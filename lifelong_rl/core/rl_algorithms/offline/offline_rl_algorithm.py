import gtimer as gt
import numpy as np
import swanlab
import abc

from lifelong_rl.core import logger
from lifelong_rl.core.rl_algorithms.rl_algorithm import _get_step_timings
from lifelong_rl.util import eval_util


class OfflineRLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            evaluation_policy,
            evaluation_env,
            evaluation_data_collector,
            replay_buffer,
            batch_size,
            max_path_length,
            num_epochs=None,
            num_eval_steps_per_epoch=None,
            num_eval_episodes=None,
            num_trains_per_train_loop=1,
            num_train_loops_per_epoch=1,
            save_snapshot_freq=1000,
            total_steps=None,
            eval_interval_steps=5000,
    ):
        self.trainer = trainer
        self.eval_policy = evaluation_policy
        self.eval_env = evaluation_env
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer

        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.save_snapshot_freq = save_snapshot_freq

        if num_eval_steps_per_epoch is None and num_eval_episodes is not None:
            self.num_eval_steps_per_epoch = num_eval_episodes * self.max_path_length
        else:
            self.num_eval_steps_per_epoch = num_eval_steps_per_epoch or 0

        self.total_steps = total_steps if total_steps is not None else (
            self.num_epochs * self.num_train_loops_per_epoch * self.num_trains_per_train_loop)
        self.eval_interval_steps = max(1, eval_interval_steps)

        self._start_step = 0
        self.post_epoch_funcs = []
        self._last_eval_iteration = None

    def _train(self):
        total_steps = int(self.total_steps)
        self.training_mode(True)
        for step in range(self._start_step, total_steps):
            train_data, indices = self.replay_buffer.random_batch(
                self.batch_size, return_indices=True)
            self.trainer.train(train_data, indices)

            is_last_step = (step + 1) == total_steps
            if (step + 1) % self.eval_interval_steps == 0 or is_last_step:
                if hasattr(self.trainer, 'log_alpha'):
                    curr_alpha = self.trainer.log_alpha.exp()
                else:
                    curr_alpha = None

                if self.num_eval_steps_per_epoch > 0:
                    self.eval_data_collector.collect_new_paths(
                        max_path_length=self.max_path_length,
                        num_samples=self.num_eval_steps_per_epoch,
                        discard_incomplete_paths=True,
                        alpha=curr_alpha,
                    )
                    self._last_eval_iteration = step + 1
                    gt.stamp('evaluation sampling', unique=False)
                else:
                    self._last_eval_iteration = None

                self.training_mode(False)
                gt.stamp('training', unique=False)
                self._end_epoch(step + 1)
                self.training_mode(True)

        self.training_mode(False)

    def train(self, start_epoch=0):
        self._start_step = start_epoch
        self._train()

    def _end_epoch(self, iteration):
        snapshot = self._get_snapshot()
        if self.save_snapshot_freq is not None and \
                iteration % self.save_snapshot_freq == 0:
            logger.save_itr_params(iteration, snapshot, prefix='offline_itr')
        gt.stamp('saving', unique=False)

        self._log_stats(iteration)

        self._end_epochs(iteration)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, iteration)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        '''
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        '''
        return snapshot

    def _end_epochs(self, epoch):
        self.eval_data_collector.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        if hasattr(self.eval_policy, 'end_epoch'):
            self.eval_policy.end_epoch(epoch)

    def _get_trainer_diagnostics(self):
        return self.trainer.get_diagnostics()

    def _get_training_diagnostics_dict(self):
        return {'policy_trainer': self._get_trainer_diagnostics()}

    # a new version of _log_stats
    def _log_stats(self, iteration):
        logger.log("Training step {} finished".format(iteration), with_timestamp=True)
        """
        Replay Buffer
        """
        logger.record_dict(self.replay_buffer.get_diagnostics(),
                           prefix='replay_buffer/')
        """
        Trainer
        """
        training_diagnostics = self._get_training_diagnostics_dict()
        for prefix in training_diagnostics:
            logger.record_dict(training_diagnostics[prefix],
                               prefix=prefix + '/')
        """
        Evaluation
        """
        # 存放所有需要被记录到 SwanLab 的指标
        log_dict_for_swanlab = dict()

        if self.num_eval_steps_per_epoch > 0 and self._last_eval_iteration == iteration:
            # 使用原有的 logger 记录
            logger.record_dict(
                self.eval_data_collector.get_diagnostics(),
                prefix='evaluation/',
            )
            eval_paths = self.eval_data_collector.get_epoch_paths()
            if hasattr(self.eval_env, 'get_diagnostics'):
                logger.record_dict(
                    self.eval_env.get_diagnostics(eval_paths),
                    prefix='evaluation/',
                )

            # --- SWANLAB MODIFICATION START ---

            # 1. 计算平均原始得分 (Raw Score)
            # get_generic_path_information 会返回一个包含 'Average Returns' 的字典
            path_info = eval_util.get_generic_path_information(eval_paths)
            logger.record_dict(path_info, prefix="evaluation/")  # 保留原有的 logger 记录

            avg_raw_score = path_info.get('Average Returns', 0)  # 从字典中获取原始分

            # 2. 计算 D4RL 标准化得分 (Normalized Score)
            normalized_score =100 * self.eval_env.get_normalized_score(avg_raw_score)

            # 3. 将得分和其他重要指标存入 SwanLab 日志字典
            log_dict_for_swanlab['Training Step'] = iteration
            log_dict_for_swanlab['Raw_Score'] = avg_raw_score
            log_dict_for_swanlab['D4RL_Normalized_Score'] = normalized_score

            # 也可以把 trainer 的损失也记录进来
            trainer_stats = self.trainer.get_diagnostics()
            for key, value in trainer_stats.items():
                # 给 key 加上前缀以防冲突
                log_dict_for_swanlab[f'trainer/{key}'] = value

            # --- SWANLAB MODIFICATION END ---

        """
        Misc
        """
        gt.stamp('logging', unique=False)
        logger.record_dict(_get_step_timings())
        logger.record_tabular('Training Step', iteration)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

        # --- SWANLAB MODIFICATION ---
        # 在函数的最后，提交所有收集到的指标到 SwanLab
        if log_dict_for_swanlab:  # 确保字典不为空
            swanlab.log(log_dict_for_swanlab)

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

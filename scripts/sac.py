from experiment_utils.launch_experiment import launch_experiment
from experiment_configs.configs.q_learning.sac_config import get_config
from experiment_configs.algorithms.offline import get_offline_algorithm
import swanlab

import argparse

def main(args):
    # Default parameters
    variant = dict(
        algorithm='SAC',
        collector_type='step',
        env_name='hopper-random-v2',
        env_kwargs=dict(),
        replay_buffer_size=int(2e6),
        reward_mean=False,  # added for easy config checking
        reward_std=-1.0,  # added for easy config checking
        policy_kwargs=dict(
            layer_size=256,
            num_q_layers=3,
            num_p_layers=3,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            policy_lr=3e-4,
            qf_lr=3e-4,
            use_automatic_entropy_tuning=True,
            policy_eval_start=0,
            num_qs=10,
            target_update_period=1,
            max_q_backup=False,
            deterministic_backup=False,
            eta=-1.0,
            use_density_penalty=False,
            flow_model_path='',
            density_random_num_samples=10,
            density_base_weight=1.0,
            density_ucb_ratio=0.0,
            density_nll_clip_min=0.0,
            density_nll_clip_max=5.0,
        ),
        offline_kwargs=dict(
            total_steps=int(1e6),
            eval_interval_steps=5000,
            max_path_length=1000,
            num_eval_episodes=10,
            batch_size=256,
            save_snapshot_freq=int(1e6),  # save final checkpoint by default
        ),
    )

    experiment_kwargs = dict(
        exp_postfix='',
        use_gpu=True,
        log_to_tensorboard=False,
    )

    # Variant
    variant['env_name'] = args.env_name
    variant['seed'] = args.seed

    variant['offline_kwargs']['total_steps'] = args.total_steps
    variant['offline_kwargs']['save_snapshot_freq'] = args.total_steps

    # SAC-N
    variant['trainer_kwargs']['policy_lr'] = args.plr
    variant['trainer_kwargs']['qf_lr'] = args.qlr

    variant['trainer_kwargs']['num_qs'] = args.num_qs
    variant['trainer_kwargs']['max_q_backup'] = args.max_q_backup
    variant['trainer_kwargs']['deterministic_backup'] = args.deterministic_backup
    variant['trainer_kwargs']['use_density_penalty'] = args.use_density_penalty
    variant['trainer_kwargs']['flow_model_path'] = args.flow_model_path
    variant['trainer_kwargs']['density_random_num_samples'] = args.density_random_num_samples
    variant['trainer_kwargs']['density_base_weight'] = args.density_base_weight
    variant['trainer_kwargs']['density_ucb_ratio'] = args.density_ucb_ratio
    variant['trainer_kwargs']['density_nll_clip_min'] = args.density_nll_clip_min
    variant['trainer_kwargs']['density_nll_clip_max'] = args.density_nll_clip_max

    variant['reward_mean'] = args.reward_mean
    variant['reward_std'] = args.reward_std
    
    # EDAC
    variant['trainer_kwargs']['eta'] = args.eta

    # experiment name
    experiment_kwargs['exp_postfix'] = ''
    
    exp_postfix = '_{}'.format(args.num_qs)
    
    exp_postfix += '_plr{:.4f}_qlr{:.4f}'.format(args.plr, args.qlr)
    if variant['trainer_kwargs']['max_q_backup']:
        exp_postfix += '_maxq'
    if variant['trainer_kwargs']['deterministic_backup']:
        exp_postfix += '_detq'
    if args.eta > 0:
        exp_postfix += '_eta{:.2f}'.format(args.eta)
    if args.reward_mean:
        exp_postfix += '_mean'
    if args.reward_std > 0:
        exp_postfix += '_std'
    if args.use_density_penalty:
        exp_postfix += '_dens'

    experiment_kwargs['exp_postfix'] = exp_postfix

    experiment_kwargs['data_args'] = {
        'reward_mean': args.reward_mean,
        'reward_std': args.reward_std,
    }
    swanlab.init(
        project="DUR-max",  # 您可以自定义项目名称
        name=f"{variant['env_name']}",  # 使用环境名和参数后缀作为实验名
        config=variant  # 将所有超参数配置传入
    )


    # Launch experiment
    launch_experiment(variant=variant,
                      get_config=get_config,
                      get_offline_algorithm=get_offline_algorithm,
                      **experiment_kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Variant
    parser.add_argument('-e',
                        '--env_name',
                        default='halfcheetah-random-v2',
                        type=str)
    parser.add_argument('--seed', default=0, type=int)
    # Misc arguments
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--log_to_tensorboard', action='store_true')
    parser.add_argument("--total_steps", default=int(1e6), type=int,
                        help='total number of gradient updates during offline training')
    # SAC
    parser.add_argument("--plr",
                        default=3e-4,
                        type=float,
                        help='policy learning rate')
    parser.add_argument("--qlr",
                        default=3e-4,
                        type=float,
                        help='Q learning rate')
    parser.add_argument("--num_qs",
                        default=10,
                        type=int,
                        help='number of Q-functions to be used')
    parser.add_argument('--max_q_backup',
                        action='store_true',
                        help='use max q backup')
    parser.add_argument('--deterministic_backup',
                        action='store_true',
                        help='use deterministic backup')
    parser.add_argument('--eta',
                        default=-1.0,
                        type=float,
                        help='eta for diversifying Q-ensemble. < 0 for SAC-N.')
    
    # reward preprocessing
    parser.add_argument("--reward_mean",
                        action='store_true',
                        help='normalize rewards to 0 mean')
    parser.add_argument("--reward_std",
                        action='store_true',
                        help='normalize rewards to 1 std')

    # density penalty
    parser.add_argument('--use_density_penalty',
                        action='store_true',
                        help='enable flow-based density penalty on Q targets')
    parser.add_argument('--flow_model_path',
                        default='',
                        type=str,
                        help='path to pretrained ConditionalFlow checkpoint')
    parser.add_argument('--density_random_num_samples',
                        default=10,
                        type=int,
                        help='number of random samples for density penalty (reserved)')
    parser.add_argument('--density_base_weight',
                        default=1.0,
                        type=float,
                        help='base weight for density penalty schedule')
    parser.add_argument('--density_ucb_ratio',
                        default=0.0,
                        type=float,
                        help='scaling factor for UCB-based penalty')
    parser.add_argument('--density_nll_clip_min',
                        default=0.0,
                        type=float,
                        help='lower clip for negative log-likelihood from flow model')
    parser.add_argument('--density_nll_clip_max',
                        default=5.0,
                        type=float,
                        help='upper clip for negative log-likelihood from flow model')

    args = parser.parse_args()

    main(args)
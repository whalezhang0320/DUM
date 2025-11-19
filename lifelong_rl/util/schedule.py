"""Basic scheduling utilities reused for density penalty weighting."""


def linear_interpolation(left: float, right: float, alpha: float) -> float:
    return left + alpha * (right - left)


class ConstantSchedule:
    def __init__(self, value: float) -> None:
        self._value = value

    def value(self, _: int) -> float:
        return self._value


class PiecewiseSchedule:
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, t: int) -> float:
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        assert self._outside_value is not None
        return self._outside_value


class LinearSchedule:
    def __init__(self, schedule_timesteps: int, final_p: float, initial_p: float = 1.0) -> None:
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t: int) -> float:
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


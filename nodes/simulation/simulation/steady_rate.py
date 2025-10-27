import time


class SteadyRate:
    """Maintains the steady cycle rate provided on initialization by adaptively sleeping an amount
    of time to make up the remaining cycle time after work is done.

    More info: https://forums.developer.nvidia.com/t/real-time-factor-in-issac-sim/225784

    Usage:

    rate = SteadyRate(rate_hz=60.)
    while True:
      app.update() # render/app update call here
      rate.sleep()  # Sleep for the remaining cycle time.

    """

    def __init__(self, rate_hz):
        self.rate_hz = rate_hz
        self.dt = 1.0 / rate_hz
        self.last_sleep_end = time.time()

    def sleep(self):
        work_elapse = time.time() - self.last_sleep_end
        sleep_time = self.dt - work_elapse
        if sleep_time > 0.0:
            time.sleep(sleep_time)
        self.last_sleep_end = time.time()
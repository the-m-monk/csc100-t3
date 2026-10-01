from csc100_t3.mjsim import tinterface as ti
from csc100_t3.model import aux


def calculate_reward(
    state: ti.StepState,
    next_state: ti.StepState,
    course: ti.CourseState,
    action: aux.DogModelAction,
    is_navigation_keepout_respected,
):
    return 0.0

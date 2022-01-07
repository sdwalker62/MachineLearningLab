import gym


def test_gym() -> bool:
    space = gym.spaces.Discrete(8)  # Set with 8 elements {0, 1, 2, ..., 7}
    x = space.sample()
    if space.contains(x) and space.n == 8:
        return True
    else:
        return False

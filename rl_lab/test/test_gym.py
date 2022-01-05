def test_gym():
    space = gym.spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
    x = space.sample()
    assert space.contains(x)
    assert space.n == 8
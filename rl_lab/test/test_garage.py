def test_garage():
    x = garage.np.flatten_tensors([numpy.ndarray([1]), numpy.ndarray([1])])
    assert len(x) == 2
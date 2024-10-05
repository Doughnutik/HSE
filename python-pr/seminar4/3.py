import pytest

@pytest.mark.parametrize("a, b, expected", [pytest.param(5, 7, 12, id='first')])
def test_add(a, b, expected):
    assert a + b == expected
    
    
@pytest.fixture
def sample_data():
    return {"a": 10, "b": 5}
    
     
def test_example(sample_data):
    assert sample_data["a"] == 10
    

@pytest.mark.sip(reason="Temporary skipped")
def test_temp_skip():
    assert 1 == 1
    
@pytest.mark.xfail
def test_known_failure():
    assert 1 / 0 == 1
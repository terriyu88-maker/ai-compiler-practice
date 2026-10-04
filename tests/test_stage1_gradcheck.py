from src.stage1_lenet.manual_backprop import run_gradcheck


def test_manual_backprop_matches_autograd():
    report = run_gradcheck(batch_size=8)
    assert report["forward_max_abs_err"] < 1e-4
    assert report["loss_abs_err"] < 1e-4
    assert report["grad_max_abs_err"] < 1e-4

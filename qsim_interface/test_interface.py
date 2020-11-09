import numpy as np
import build.qsim_interface as pbi

ComplexType = np.dtype('complex64')


def test_sampler_setters():
  sampler_cpp = pbi.Sampler(3, True)
  sampler_cpp.set_random_seed(22)
  sampler_cpp.set_initial_registers({"a": 1})
  sampler_cpp.set_register_order(["c", "d", "e"])

  # state_vec = np.arange((2 ** max_qubits_)).astype(ComplexType)
  axes = ["a", "b", "c"]
  state_vec = np.zeros((2 ** len(axes)), ComplexType)
  state_vec[3] = 1.0
  state_vec[5] = 2.0
  state_vec[0] = 3.0

  sampler_cpp.bind_initial_state(state_vec, axes)


def test_add_kop():
  sampler_cpp = pbi.Sampler(3, True)

  channels = {
      (0,): [[np.eye(2, dtype=ComplexType).ravel(), [], ["a"], [], [], []]]}
  cond_regs = ("conditional register",)
  is_recorded = False
  is_virtual = False
  label = "ID_a"

  sampler_cpp.add_koperation(channels, cond_regs, is_recorded, label,
                             is_virtual)


def test_add_cop():
  sampler_cpp = pbi.Sampler(3, True)

  copy_data = {(0,): (0,),
               (1,): (1,)}
  copy_flip = {(0,): (1,),
               (1,): (0,)}

  channels_map = {(0,): ([[copy_data, ("a",), ("b",)],
                          [copy_flip, ("a",), ("b",)]], [.8, .2]),
                  (1,): ([[copy_data, ("a",), ("b",)],
                          [copy_flip, ("a",), ("b",)]], [.6, .4])
                  }

  sampler_cpp.add_coperation(channels_map, ("c",), {"b"}, False)


def test_samples():
  sampler_cpp = pbi.Sampler(3, True)
  sampler_cpp.set_random_seed(11)
  cond_reg = "R"
  sampler_cpp.set_initial_registers({cond_reg: 0})
  labels = [cond_reg]

  # state_vec = np.arange((2 ** max_qubits_)).astype(ComplexType)
  state_vec = np.zeros((2 ** 3,), ComplexType)
  state_vec[0] = 1.0
  axes = ["a", "b", "c"]
  sampler_cpp.bind_initial_state(state_vec, axes)

  sqrt_half = np.sqrt(0.5)
  channels = {
      (0,): [
          [sqrt_half * np.eye(2, dtype=ComplexType).ravel(), [], ["a"], [], [],
           []],
          [sqrt_half * np.array([0, 1, 1, 0], ComplexType), [], ["a"], [], [],
           []]
      ]}
  cond_regs = (cond_reg,)
  is_recorded = True
  is_virtual = False
  flip_a = "flip_a"
  labels.append(flip_a)

  sampler_cpp.add_koperation(channels, cond_regs, is_recorded, flip_a,
                             is_virtual)

  # flip b if a was flipped
  channels = {
      (0,): [[np.eye(2, dtype=ComplexType).ravel(), [], ["b"], [], [], []]],
      (1,): [[np.array([0, 1, 1, 0], ComplexType).ravel(), [], ["b"], [], [],
              []]]}  #
  cond_regs = (flip_a,)
  is_recorded = False
  is_virtual = False

  sampler_cpp.add_koperation(channels, cond_regs, is_recorded, 'unlabelled',
                             is_virtual)

  channels = {
      (0,): [
          [np.array([1, 0, 0, 0], ComplexType), [], ["a"], [], [], ["a"]],
          [np.array([0, 1, 0, 0], ComplexType), [], ["a"], [], [], ["a"]]
      ]}
  cond_regs = (cond_reg,)
  is_recorded = True
  is_virtual = False
  label = "measure_a"
  labels.append(label)

  sampler_cpp.add_koperation(channels, cond_regs, is_recorded, label,
                             is_virtual)

  sampler_cpp.set_register_order(labels)

  reg_mat, out_arrays, axis_orders = sampler_cpp.sample_states(10)
  out_arrays = np.array([arr.view(ComplexType) for arr in out_arrays])

  assert reg_mat.shape == (10, 3)
  # expect second and third columns to always agree, first column to always
  # be zero
  assert np.array_equal(reg_mat[:, 0], np.zeros_like(reg_mat[:, 0]))
  assert np.array_equal(reg_mat[:, 1], reg_mat[:, 2])

  # b was flipped exactly when measure_a and flip_a are true
  assert axis_orders == [["b", "c"]]
  state_00 = np.array([1, 0, 0, 0], ComplexType)
  state_10 = np.array([0, 0, 1, 0], ComplexType)
  for final_arr, measure_outcome in zip(out_arrays, reg_mat[:, 1]):
    if measure_outcome:
      np.testing.assert_array_equal(final_arr, state_10)
    else:
      np.testing.assert_array_equal(final_arr, state_00)


def test_samples_1():
  for _ in range(100):
    sampler_cpp = pbi.Sampler(1, True)

    sampler_cpp.set_initial_registers({})

    # prepare "c" in 0 state
    channels = {
        (): [[np.array([1, 0, 0, 0], ComplexType), ["c"], ["c"], [], [], []]]
    }

    sampler_cpp.add_koperation(channels, (), False, "prep c", False)

    sampler_cpp.set_register_order(())

    state_vec = np.zeros((4,), ComplexType)
    state_vec[3] = 1.0
    axes = ["a", "b"]
    sampler_cpp.bind_initial_state(state_vec, axes)

    reg_mat, out_arrays, axis_orders = sampler_cpp.sample_states(1)
    out_arrays = np.array([arr.view(ComplexType) for arr in out_arrays])

    expected_arr = np.zeros((8,), ComplexType)
    expected_arr[3] = 1
    for arr in out_arrays:
      np.testing.assert_array_equal(arr, expected_arr)

    assert axis_orders[0] == ["c", "a", "b"]
    assert reg_mat.size == 0


def test_state_prep():
  for _ in range(100):
    sampler_cpp = pbi.Sampler(1, True)
    sampler_cpp.set_random_seed(11)

    sampler_cpp.set_initial_registers({})

    # prepare "d" in 0 state
    channels = {
        (): [[np.array([1, 0, 0, 0], ComplexType), ["M0"], ["M0"], [], [], []]]
    }

    sampler_cpp.add_koperation(channels, (), False, "prep M0", False)

    sampler_cpp.set_register_order(())

    state_vec = np.zeros((8,), ComplexType)
    state_vec[1] = 1.0
    axes = ["D0", "D1", "D2"]
    sampler_cpp.bind_initial_state(state_vec, axes)

    reg_mat, out_arrays, axis_orders = sampler_cpp.sample_states(1)
    out_arrays = np.array([arr.view(ComplexType) for arr in out_arrays])
    assert not np.any(np.isnan(out_arrays[0]))

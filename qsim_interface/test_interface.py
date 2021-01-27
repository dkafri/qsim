from itertools import product

import numpy as np
import pytest
import qsim_kraus_sim as pbi

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
        (): [[np.array([1, 0,
                        0, 0], ComplexType), ["c"], ["c"], [], [], []]]
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


def test_state_prep_and_removal_repeated():
  sampler_cpp = pbi.Sampler(2, True)
  sampler_cpp.set_random_seed(11)

  sampler_cpp.set_initial_registers({})

  # prepare "d" in 0 state

  channel_prep = {
      (): [[np.array([1, 0,
                      0, 0], ComplexType), ["M0"], ["M0"], [], [], []]]
  }

  # destructively measure "d"
  channel_meas = {
      (): [[np.array([1, 0,
                      0, 0], ComplexType), [], ["M0"], [], [], ["M0"]],
           [np.array([0, 1,
                      0, 0], ComplexType), [], ["M0"], [], [], ["M0"]]
           ]
  }

  m_label_0 = "M0-0"
  m_label_1 = "M0-1"
  sampler_cpp.add_koperation(channel_prep, (), False, "prep M0", False)
  sampler_cpp.add_koperation(channel_meas, (), True, m_label_0, False)
  sampler_cpp.add_koperation(channel_prep, (), False, "prep M0", False)
  sampler_cpp.add_koperation(channel_meas, (), True, m_label_1, False)

  sampler_cpp.set_register_order((m_label_0, m_label_1))

  state_vec = np.zeros((1,), ComplexType)
  state_vec[0] = 1.0
  axes = []
  sampler_cpp.bind_initial_state(state_vec, axes)

  reg_mat, out_arrays, axis_orders = sampler_cpp.sample_states(2)
  out_arrays = np.array([arr.view(ComplexType) for arr in out_arrays])
  assert not np.any(np.isnan(out_arrays[0]))
  np.testing.assert_array_equal(reg_mat, np.zeros_like(reg_mat))


def test_multiply_by_scalar():
  sampler_cpp = pbi.Sampler(1, True)
  sampler_cpp.set_random_seed(11)

  sampler_cpp.set_initial_registers({})

  # Trivially sample from a 1-element matrix, expect second outcome
  channels = {
      (): [[np.array([0], ComplexType), [], [], [], [], []],
           [np.array([1], ComplexType), [], [], [], [], []]]
  }

  m_label = "trivial sample"
  sampler_cpp.add_koperation(channels, (), True, m_label, False)

  sampler_cpp.set_register_order((m_label,))

  state_vec = np.zeros((8,), ComplexType)
  state_vec[1] = 1.0
  axes = ["D0", "D1", "D2"]
  sampler_cpp.bind_initial_state(state_vec, axes)

  reg_mat, out_arrays, axis_orders = sampler_cpp.sample_states(1)

  np.testing.assert_array_equal(reg_mat, [[1]])


# Note: the exception is only thrown if the backend is compiled in debug mode.
def test_add_virtual_register_twice_throws_error():
  sampler_cpp = pbi.Sampler(3, True)

  sampler_cpp.set_initial_registers({"a": 0, "c": 1})

  copy_data = {(0,): (0,),
               (1,): (1,)}
  copy_flip = {(0,): (1,),
               (1,): (0,)}

  # Virtual operaton that, conditioned on 'c', applies either a copy or a flip
  # copy from a to a new register b.
  channels_map = {(0,): ([[copy_data, ("a",), ("b",)],
                          [copy_flip, ("a",), ("b",)]], [.8, .2]),
                  (1,): ([[copy_data, ("a",), ("b",)],
                          [copy_flip, ("a",), ("b",)]], [.6, .4])
                  }

  sampler_cpp.add_coperation(channels_map, ("c",), {"b"}, True)

  # Concretely copy c to a
  channels_map = {(): ([[copy_data, ("c",), ("a",)]], [1.0])}
  sampler_cpp.add_coperation(channels_map, (), set(), False)

  # Repeat the original virtual operation that again creates b
  channels_map = {(0,): ([[copy_data, ("a",), ("b",)],
                          [copy_flip, ("a",), ("b",)]], [.8, .2]),
                  (1,): ([[copy_data, ("a",), ("b",)],
                          [copy_flip, ("a",), ("b",)]], [.6, .4])
                  }

  sampler_cpp.add_coperation(channels_map, ("c",), {"b"}, True)

  sampler_cpp.set_register_order(('a', 'c'))

  # Implement sampling
  state_vec = np.zeros((8,), ComplexType)
  state_vec[1] = 1.0
  axes = ["D0", "D1", "D2"]
  sampler_cpp.bind_initial_state(state_vec, axes)

  with pytest.raises(RuntimeError):
    reg_mat, out_arrays, axis_orders = sampler_cpp.sample_states(1)


def test_leaky_cz_rpa_example():
  sampler_cpp = pbi.Sampler(1, False)

  # Leaky CZ where 11 state is always incoherently mapped to and from 02.
  q0 = 'a'
  q1 = 'b'
  leaky_cz_channels = {
      (0, 0): [([1., 0., 0., 0.,
                 0., 1., 0., 0.,
                 0., 0., 1., 0.,
                 0., 0., 0., 0.],
                [],
                [q0, q1],
                [],
                [],
                []),
               ([0., 0., 0., np.sqrt(1 / 2),
                 0., 0., 0., 0.,
                 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [],
                [q0, q1],
                [],
                [],
                [q1]),
               ([0., 0., 0., 0. - np.sqrt(1 / 2) * 1j,
                 0., 0., 0., 0.,
                 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [],
                [q0, q1],
                [],
                [],
                [q1])],
      (0, 1): [([0., 0.,
                 0., 1.],
                [],
                [q0],
                [],
                [],
                []),
               ([0., 0., 0., 0.,
                 0., 0., 0., 0.,
                 0., 0., 0., 0.,
                 np.sqrt(1 / 2), 0., 0., 0.],
                [q1],
                [q0, q1],
                [],
                [],
                []),
               ([0., 0., 0., 0.,
                 0., 0., 0., 0.,
                 0., 0., 0., 0.,
                 np.sqrt(1 / 2) * 1j, 0., 0., 0.],
                [q1],
                [q0, q1],
                [],
                [],
                [])],
      (1, 0): [([1., 0.,
                 0., 1.],
                [],
                [q1],
                [],
                [],
                [])],
      (1, 1): [([1.], [], [], [], [], [])]}
  a_leaked = 'a leaked'
  b_leaked = 'b leaked'
  leaky_cz_args = (
      leaky_cz_channels, (a_leaked, b_leaked), True, 'leaky CZ', False)

  table = np.zeros((2, 2, 3, 2), int)
  table[0, 0, 1:] = (0, 1)  # cc -> cl
  table[0, 1, 0] = (0, 1)  # cl ->cl
  table[1, 0, 0] = (1, 0)  # lc ->lc
  table[1, 1, 0] = (1, 1)  # lc ->ll
  update_map = {(): ([({s: tuple(table[s]) for s in
                        product(range(2), range(2), range(3))},
                       (a_leaked, b_leaked, 'leaky CZ'),
                       (a_leaked, b_leaked))],
                     np.array([1.]))}
  update_args = (update_map, (), set(), False)

  sampler_cpp.add_koperation(*leaky_cz_args)
  sampler_cpp.add_coperation(*update_args)
  sampler_cpp.add_koperation(*leaky_cz_args)
  sampler_cpp.add_coperation(*update_args)
  sampler_cpp.add_koperation(*leaky_cz_args)
  sampler_cpp.add_coperation(*update_args)

  sampler_cpp.bind_initial_state([0., 0., 0., 1.],
                                 (q0, q1))
  init_reg = {a_leaked: 0, b_leaked: 0}
  sampler_cpp.set_initial_registers(init_reg)
  sampler_cpp.set_random_seed(13)
  sampler_cpp.set_register_order((a_leaked, b_leaked))

  reg_mat, out_arrays, axis_orders = sampler_cpp.sample_states(10)
  expected = np.zeros_like(reg_mat)
  expected[:, 1] = 1
  np.testing.assert_array_equal(reg_mat, expected)

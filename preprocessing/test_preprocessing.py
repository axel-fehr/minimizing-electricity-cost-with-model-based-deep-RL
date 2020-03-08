import unittest
import numpy as np
from numpy import testing as npt
from preprocessing import preprocessing as prep

class TestTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_z_scores_with_means_and_stds(self):
        test_array = np.array([[1,10], [4,14]])
        normed_arr, means, std = prep.get_z_scores_with_means_and_stds(test_array)
        npt.assert_array_equal(normed_arr, np.array([[-1,-1], [1,1]]))
        npt.assert_array_equal(means, np.array([2.5, 12]))
        npt.assert_array_equal(std, np.array([1.5, 2]))

    def test_get_corresponding_time_stamp_string(self):
        time_stamp = "11.07.2016 11:13"
        distance_between_time_stamps_in_mins = 30
        result = prep.get_corresponding_time_stamp_string(time_stamp, distance_between_time_stamps_in_mins)
        self.assertEqual(result, "11.07.2016 11:43")

        time_stamp = "10.09.2018 07:03"
        distance_between_time_stamps_in_mins = 60
        result = prep.get_corresponding_time_stamp_string(time_stamp, distance_between_time_stamps_in_mins)
        self.assertEqual(result, "10.09.2018 08:03")

        time_stamp = "10.09.2018 07:03"
        distance_between_time_stamps_in_mins = 90
        result = prep.get_corresponding_time_stamp_string(time_stamp, distance_between_time_stamps_in_mins)
        self.assertEqual(result, "10.09.2018 08:33")

        time_stamp = "10.09.2018 07:03"
        distance_between_time_stamps_in_mins = 120
        result = prep.get_corresponding_time_stamp_string(time_stamp, distance_between_time_stamps_in_mins)
        self.assertEqual(result, "10.09.2018 09:03")

        time_stamp = "10.09.2018 09:35"
        distance_between_time_stamps_in_mins = 150
        result = prep.get_corresponding_time_stamp_string(time_stamp, distance_between_time_stamps_in_mins)
        self.assertEqual(result, "10.09.2018 12:05")

        time_stamp = "11.07.2016 22:59"
        distance_between_time_stamps_in_mins = 5
        result = prep.get_corresponding_time_stamp_string(time_stamp, distance_between_time_stamps_in_mins)
        self.assertEqual(result, "11.07.2016 23:04")

        time_stamp = "11.07.2016 14:34"
        distance_between_time_stamps_in_mins = 26
        result = prep.get_corresponding_time_stamp_string(time_stamp, distance_between_time_stamps_in_mins)
        self.assertEqual(result, "11.07.2016 15:00")

        time_stamp = "11.07.2016 23:45"
        distance_between_time_stamps_in_mins = 20
        result = prep.get_corresponding_time_stamp_string(time_stamp, distance_between_time_stamps_in_mins)
        self.assertEqual(result, "12.07.2016 00:05")

        time_stamp = "30.11.2016 23:50"
        distance_between_time_stamps_in_mins = 15
        result = prep.get_corresponding_time_stamp_string(time_stamp, distance_between_time_stamps_in_mins)
        self.assertEqual(result, "01.12.2016 00:05")

        time_stamp = "31.10.2016 16:14"
        distance_between_time_stamps_in_mins = 12 * 60
        result = prep.get_corresponding_time_stamp_string(time_stamp, distance_between_time_stamps_in_mins)
        self.assertEqual(result, "01.11.2016 04:14")

        time_stamp = "04.12.2016 23:50"
        distance_between_time_stamps_in_mins = 24 * 60
        result = prep.get_corresponding_time_stamp_string(time_stamp, distance_between_time_stamps_in_mins)
        self.assertEqual(result, "05.12.2016 23:50")

        time_stamp = "31.12.2016 00:00"
        distance_between_time_stamps_in_mins = 24 * 60
        result = prep.get_corresponding_time_stamp_string(time_stamp, distance_between_time_stamps_in_mins)
        self.assertEqual(result, "01.01.2017 00:00")

        time_stamp = "31.12.2016 13:42"
        distance_between_time_stamps_in_mins = 24 * 60
        result = prep.get_corresponding_time_stamp_string(time_stamp, distance_between_time_stamps_in_mins)
        self.assertEqual(result, "01.01.2017 13:42")

        time_stamp = "31.12.2018 23:59"
        distance_between_time_stamps_in_mins = 1
        result = prep.get_corresponding_time_stamp_string(time_stamp, distance_between_time_stamps_in_mins)
        self.assertEqual(result, "01.01.2019 00:00")

        time_stamp = "31.12.2018 23:57"
        distance_between_time_stamps_in_mins = 5
        result = prep.get_corresponding_time_stamp_string(time_stamp, distance_between_time_stamps_in_mins)
        self.assertEqual(result, "01.01.2019 00:02")

    def test_is_later_time_stamp(self):
        self.assertTrue(prep.is_later_time_stamp('12.11.2016 13:43', '12.11.2016 13:42'))
        self.assertTrue(prep.is_later_time_stamp('11.12.2017 13:42', '12.11.2016 13:42'))
        self.assertTrue(prep.is_later_time_stamp('11.12.2016 01:40', '12.11.2016 13:42'))
        self.assertTrue(prep.is_later_time_stamp('12.11.2016 16:00', '12.11.2016 15:59'))
        self.assertTrue(prep.is_later_time_stamp('13.11.2016 00:00', '12.11.2016 23:59'))
        self.assertTrue(prep.is_later_time_stamp('01.12.2016 00:00', '30.11.2016 23:59'))
        self.assertFalse(prep.is_later_time_stamp('12.11.2016 13:41', '12.11.2016 13:42'))
        self.assertFalse(prep.is_later_time_stamp('12.11.2016 13:42', '12.11.2016 13:42'))
        self.assertFalse(prep.is_later_time_stamp('30.11.2016 23:59', '01.12.2016 00:00'))
        self.assertFalse(prep.is_later_time_stamp('12.11.2016 15:59', '31.12.2016 23:42'))

    def test_get_state_transitions_as_separate_arrays(self):
        state_action_pair_feature_order = ['generation_in_W', 'load_in_W', 'charge_rate_in_W']
        temporal_resolution_in_mins = 1
        merged_data_with_time_stamps = [[1.5, 2.4, 4.6, '12.11.2016 13:42'],
                                        [7.3, 1.2, 8.2, '12.11.2016 13:43']]
        a, b, time_stamps = prep.get_state_transitions_as_separate_arrays(merged_data_with_time_stamps,
                                                                          temporal_resolution_in_mins,
                                                                          state_action_pair_feature_order)
        a_des = np.array([[1.5, 2.4, 4.6]])
        b_des = np.array([[7.3, 1.2]])
        npt.assert_array_equal(a, a_des)
        npt.assert_array_equal(b, b_des)

        merged_data_with_time_stamps = [[1.5, 2.4, 7.6, '12.11.2016 13:42'],
                                        [4.3, 9.9, 1.2, '12.11.2016 13:43'],
                                        [2.6, 5.5, 3.0, '12.11.2016 13:45'],
                                        [8.1, 3.6, 0.8, '12.11.2016 13:46']]
        a, b, time_stamps = prep.get_state_transitions_as_separate_arrays(merged_data_with_time_stamps,
                                                                          temporal_resolution_in_mins,
                                                                          state_action_pair_feature_order)
        a_des = np.array([[1.5, 2.4, 7.6], [2.6, 5.5, 3.0]])
        b_des = np.array([[4.3, 9.9], [8.1, 3.6]])
        npt.assert_array_equal(a, a_des)
        npt.assert_array_equal(b, b_des)

        state_action_pair_feature_order = ['generation_in_W', 'charge_rate_in_W', 'load_in_W']
        merged_data_with_time_stamps = [[1.5, 2.4, 7.6, '12.11.2016 13:42'],
                                        [4.3, 9.9, 1.2, '12.11.2016 13:43'],
                                        [2.6, 5.5, 3.0, '12.11.2016 13:45'],
                                        [8.1, 3.6, 0.8, '12.11.2016 13:46']]
        a, b, time_stamps = prep.get_state_transitions_as_separate_arrays(merged_data_with_time_stamps,
                                                                          temporal_resolution_in_mins,
                                                                          state_action_pair_feature_order)
        a_des = np.array([[1.5, 2.4, 7.6], [2.6, 5.5, 3.0]])
        b_des = np.array([[4.3, 1.2], [8.1, 0.8]])
        npt.assert_array_equal(a, a_des)
        npt.assert_array_equal(b, b_des)

        temporal_resolution_in_mins = 5
        state_action_pair_feature_order = ['charge_rate_in_W', 'generation_in_W', 'load_in_W']
        merged_data_with_time_stamps = [[1.5, 2.4, 7.6, '12.11.2016 13:42'],
                                        [4.3, 9.9, 1.2, '12.11.2016 13:43'],
                                        [2.6, 5.5, 3.0, '12.11.2016 13:45'],
                                        [8.1, 3.6, 0.8, '12.11.2016 13:48'],
                                        [6.2, 0.9, 4.5, '12.11.2016 13:49'],
                                        [7.7, 2.6, 7.2, '12.11.2016 13:53']]
        a, b, time_stamps = prep.get_state_transitions_as_separate_arrays(merged_data_with_time_stamps,
                                                                          temporal_resolution_in_mins,
                                                                          state_action_pair_feature_order)
        a_des = np.array([[4.3, 9.9, 1.2], [8.1, 3.6, 0.8]])
        b_des = np.array([[3.6, 0.8], [2.6, 7.2]])
        npt.assert_array_equal(a, a_des)
        npt.assert_array_equal(b, b_des)

        temporal_resolution_in_mins = 60
        merged_data_with_time_stamps = [[1.5, 2.4, 7.6, '29.11.2016 23:42'],
                                        [4.3, 9.9, 1.2, '29.11.2016 23:43'],
                                        [2.6, 5.5, 3.0, '30.11.2016 00:42'],
                                        [8.1, 3.6, 0.8, '30.11.2016 23:42'],
                                        [6.2, 0.9, 4.5, '01.12.2016 00:42'],
                                        [7.7, 2.6, 7.2, '01.12.2016 01:45']]
        a, b, time_stamps = prep.get_state_transitions_as_separate_arrays(merged_data_with_time_stamps,
                                                                          temporal_resolution_in_mins,
                                                                          state_action_pair_feature_order)
        a_des = np.array([[1.5, 2.4, 7.6], [8.1, 3.6, 0.8]])
        b_des = np.array([[5.5, 3.0], [0.9, 4.5]])
        npt.assert_array_equal(a, a_des)
        npt.assert_array_equal(b, b_des)

    def test_find_idxs_of_n_longest_state_sequences(self):
        temporal_resolution_in_mins = 1
        states = np.array([[1.5, 2.4, 7.6, 6.2],
                           [4.3, 9.9, 1.2, 0.1],
                           [2.6, 5.5, 3.0, 4.0],
                           [8.1, 3.6, 0.8, 1.7],
                           [6.2, 0.9, 4.5, 5.2],
                           [7.7, 2.6, 7.2, 9.4]])
        time_stamps = ['29.11.2016 23:42',
                       '29.11.2016 23:43',
                       '30.11.2016 23:58',
                       '30.11.2016 23:59',
                       '01.12.2016 00:00',
                       '01.12.2016 00:01']
        result = prep.find_idxs_of_n_longest_state_sequences(states, time_stamps, 1, temporal_resolution_in_mins)
        self.assertEqual([[2,3,4,5]], result)

        temporal_resolution_in_mins = 5
        time_stamps = ['29.11.2016 23:42',
                       '29.11.2016 23:47',
                       '29.11.2016 23:52',
                       '29.11.2016 23:57',
                       '30.11.2016 00:02',
                       '30.11.2016 00:12']
        result = prep.find_idxs_of_n_longest_state_sequences(states, time_stamps, 1, temporal_resolution_in_mins)
        self.assertEqual([[0,1,2,3,4]], result)

        temporal_resolution_in_mins = 10
        time_stamps = ['13.11.2016 09:46',
                       '13.11.2016 09:56',
                       '13.11.2016 10:06',
                       '13.11.2016 11:16',
                       '13.11.2016 11:26',
                       '13.11.2016 11:46']
        result = prep.find_idxs_of_n_longest_state_sequences(states, time_stamps, 1, temporal_resolution_in_mins)
        self.assertEqual([[0,1,2]], result)

        temporal_resolution_in_mins = 10
        time_stamps = ['13.11.2016 09:46',
                       '13.11.2016 09:56',
                       '13.11.2016 10:06',
                       '13.11.2016 11:16',
                       '13.11.2016 11:26',
                       '13.11.2016 11:46']
        result = prep.find_idxs_of_n_longest_state_sequences(states, time_stamps, 2, temporal_resolution_in_mins)
        self.assertEqual([[0,1,2],[3,4]], result)

        temporal_resolution_in_mins = 1
        states = np.array([[1.5, 2.4, 7.6, 6.2],
                           [4.3, 9.9, 1.2, 0.1],
                           [2.6, 5.5, 3.0, 4.0],
                           [8.1, 3.6, 0.8, 1.7],
                           [6.9, 2.1, 9.2, 6.0],
                           [8.3, 5.9, 5.7, 1.9],
                           [6.2, 0.9, 4.5, 5.2],
                           [7.7, 2.6, 7.2, 9.4]])
        time_stamps = ['30.11.2016 23:42',
                       '30.11.2016 23:44',
                       '30.11.2016 23:45',
                       '30.11.2016 23:47',
                       '30.11.2016 23:58',
                       '30.11.2016 23:59',
                       '01.12.2016 00:00',
                       '01.12.2016 00:01']
        result = prep.find_idxs_of_n_longest_state_sequences(states, time_stamps, 2, temporal_resolution_in_mins)
        self.assertEqual([[4,5,6,7],[1,2]], result)

    def test_get_simulation_data(self):
        temporal_resolution_in_mins = 1
        state_action_pair_feature_order = ['charge_rate_in_W', 'generation_in_W', 'load_in_W']
        s_a_pairs = np.array([[1.5, 2.4, 7.6],
                              [4.3, 9.9, 1.2],
                              [2.6, 5.5, 3.0],
                              [8.1, 3.6, 0.8],
                              [6.2, 0.9, 4.5],
                              [7.7, 2.6, 7.2]])
        time_stamps = ['29.11.2016 23:42',
                       '29.11.2016 23:43',
                       '30.11.2016 23:58',
                       '30.11.2016 23:59',
                       '01.12.2016 00:00',
                       '01.12.2016 00:01']
        num_days_of_simulation = 3 / (24 * 60)
        sim_states, sim_time_stamps = prep.get_simulation_data(s_a_pairs, 
                                                               time_stamps,
                                                               state_action_pair_feature_order,
                                                               temporal_resolution_in_mins,
                                                               num_days_of_simulation)
        des_sim_states = np.array([[5.5, 3.0],
                                   [3.6, 0.8],
                                   [0.9, 4.5]])
        des_time_stamps = ['30.11.2016 23:58',
                           '30.11.2016 23:59',
                           '01.12.2016 00:00']
        npt.assert_array_equal(sim_states, des_sim_states)
        self.assertEqual(sim_time_stamps, des_time_stamps)

        state_action_pair_feature_order = ['generation_in_W', 'load_in_W', 'charge_rate_in_W']
        sim_states, sim_time_stamps = prep.get_simulation_data(s_a_pairs, 
                                                               time_stamps,
                                                               state_action_pair_feature_order,
                                                               temporal_resolution_in_mins,
                                                               num_days_of_simulation)
        des_sim_states = np.array([[2.6, 5.5],
                                   [8.1, 3.6],
                                   [6.2, 0.9]])
        npt.assert_array_equal(sim_states, des_sim_states)
        self.assertEqual(sim_time_stamps, des_time_stamps)

        state_action_pair_feature_order = ['charge_rate_in_W', 'generation_in_W', 'load_in_W']
        time_stamps = ['28.11.2016 13:42',
                       '29.11.2016 13:42',
                       '30.11.2016 13:42',
                       '01.12.2016 01:42',
                       '01.12.2016 13:42',
                       '02.12.2016 01:42']
        temporal_resolution_in_mins = 12 * 60
        num_days_of_simulation = 2
        sim_states, sim_time_stamps = prep.get_simulation_data(s_a_pairs, 
                                                               time_stamps,
                                                               state_action_pair_feature_order,
                                                               temporal_resolution_in_mins,
                                                               num_days_of_simulation)
        des_sim_states = np.array([[5.5, 3.0],
                                   [3.6, 0.8],
                                   [0.9, 4.5],
                                   [2.6, 7.2]])
        des_time_stamps = ['30.11.2016 13:42',
                           '01.12.2016 01:42',
                           '01.12.2016 13:42',
                           '02.12.2016 01:42']
        npt.assert_array_equal(sim_states, des_sim_states)
        self.assertEqual(sim_time_stamps, des_time_stamps)

        state_action_pair_feature_order = ['generation_in_W', 'charge_rate_in_W', 'load_in_W']
        sim_states, sim_time_stamps = prep.get_simulation_data(s_a_pairs, 
                                                               time_stamps,
                                                               state_action_pair_feature_order,
                                                               temporal_resolution_in_mins,
                                                               num_days_of_simulation)
        des_sim_states = np.array([[2.6, 3.0],
                                   [8.1, 0.8],
                                   [6.2, 4.5],
                                   [7.7, 7.2]])
        npt.assert_array_equal(sim_states, des_sim_states)
        self.assertEqual(sim_time_stamps, des_time_stamps)

    def test_get_stm_training_data(self):
        state_action_pairs = np.array([[1.5, 2.4, 7.6, 6.2],
                                       [4.3, 9.9, 1.2, 0.1],
                                       [2.6, 5.5, 3.0, 4.0],
                                       [8.1, 3.6, 0.8, 1.7],
                                       [6.2, 0.9, 4.5, 5.2],
                                       [7.7, 2.6, 7.2, 9.4]])
        resulting_states = np.array([[2.4, 7.6, 6.2],
                                     [9.9, 1.2, 0.1],
                                     [5.5, 3.0, 4.0],
                                     [3.6, 0.8, 1.7],
                                     [0.9, 4.5, 5.2],
                                     [2.6, 7.2, 9.4]])
        s_a_pairs_time_stamps = ['28.11.2016 13:42',
                                 '29.11.2016 13:42',
                                 '30.11.2016 13:42',
                                 '01.12.2016 01:42',
                                 '01.12.2016 13:42',
                                 '02.12.2016 01:42']
        simulation_time_stamps = ['30.11.2016 13:42',
                                  '01.12.2016 01:42',
                                  '01.12.2016 13:42',
                                  '02.12.2016 01:42']
        stm_x, stm_y = prep.get_stm_training_data(state_action_pairs, resulting_states, s_a_pairs_time_stamps, simulation_time_stamps)
        des_stm_x = np.array([[1.5, 2.4, 7.6, 6.2],
                              [4.3, 9.9, 1.2, 0.1]])
        des_stm_y = np.array([[2.4, 7.6, 6.2],
                              [9.9, 1.2, 0.1]])
        npt.assert_array_equal(stm_x, des_stm_x)
        npt.assert_array_equal(stm_y, des_stm_y)

        state_action_pairs = np.array([[1.5, 2.4, 7.6],
                                       [4.3, 9.9, 1.2],
                                       [2.6, 5.5, 3.0],
                                       [8.1, 3.6, 0.8],
                                       [6.2, 0.9, 4.5],
                                       [7.7, 2.6, 7.2]])
        resulting_states = np.array([[2.4, 7.6],
                                     [9.9, 1.2],
                                     [5.5, 3.0],
                                     [3.6, 0.8],
                                     [0.9, 4.5],
                                     [2.6, 7.2]])
        s_a_pairs_time_stamps = ['29.11.2016 23:43',
                                 '30.11.2016 23:58',
                                 '30.11.2016 23:59',
                                 '01.12.2016 00:00',
                                 '01.12.2016 00:01',
                                 '01.12.2016 00:03']
        simulation_time_stamps = ['30.11.2016 23:58',
                                  '30.11.2016 23:59',
                                  '01.12.2016 00:00']
        stm_x, stm_y = prep.get_stm_training_data(state_action_pairs, resulting_states, s_a_pairs_time_stamps, simulation_time_stamps)
        des_stm_x = np.array([[1.5, 2.4, 7.6],
                              [6.2, 0.9, 4.5],
                              [7.7, 2.6, 7.2]])
        des_stm_y = np.array([[2.4, 7.6],
                              [0.9, 4.5],
                              [2.6, 7.2]])
        npt.assert_array_equal(stm_x, des_stm_x)
        npt.assert_array_equal(stm_y, des_stm_y)

    def test_replace_missing_data_with_interpolations(self):
        data_with_time_stamps = [[1.2, 2.5, 7.6, '04.11.2016 11:56'],
                                 [4.4, 9.9, 1.2, '04.11.2016 11:58'],
                                 [2.6, 5.5, 3.0, '04.11.2016 11:59'],
                                 [7.7, 2.6, 7.2, '04.11.2016 12:00']]
        result = prep.replace_missing_data_with_interpolations(data_with_time_stamps)
        result_values = [x[:-1] for x in result]
        result_time_stamps = [x[-1] for x in result]
        des_values = np.array([[1.2, 2.5, 7.6],
                               [2.8, 6.2, 4.4],
                               [4.4, 9.9, 1.2],
                               [2.6, 5.5, 3.0],
                               [7.7, 2.6, 7.2]])
        des_time_stamps = ['04.11.2016 11:56',
                           '04.11.2016 11:57',
                           '04.11.2016 11:58',
                           '04.11.2016 11:59',
                           '04.11.2016 12:00']
        npt.assert_almost_equal(np.asarray(result_values), des_values)
        self.assertEqual(result_time_stamps, des_time_stamps)

        data_with_time_stamps = [[1.2, 2.5, 7.6, '04.11.2016 11:56'],
                                 [4.4, 9.9, 1.2, '04.11.2016 11:58'],
                                 [2.6, 5.5, 3.0, '04.11.2016 12:00'],
                                 [7.7, 2.6, 7.2, '04.11.2016 12:01']]
        result = prep.replace_missing_data_with_interpolations(data_with_time_stamps)
        result_values = [x[:-1] for x in result]
        result_time_stamps = [x[-1] for x in result]
        des_values = np.array([[1.2, 2.5, 7.6],
                               [2.8, 6.2, 4.4],
                               [4.4, 9.9, 1.2],
                               [3.5, 7.7, 2.1],
                               [2.6, 5.5, 3.0],
                               [7.7, 2.6, 7.2]])
        des_time_stamps = ['04.11.2016 11:56',
                           '04.11.2016 11:57',
                           '04.11.2016 11:58',
                           '04.11.2016 11:59',
                           '04.11.2016 12:00',
                           '04.11.2016 12:01']
        npt.assert_almost_equal(np.asarray(result_values), des_values)
        self.assertEqual(result_time_stamps, des_time_stamps)

        data_with_time_stamps = [[1.2, 2.5, 7.6, '14.12.2016 23:55'],
                                 [6.1, 7.3, 8.5, '14.12.2016 23:56'],
                                 [3.1, 9.1, 1.6, '14.12.2016 23:59'],
                                 [2.9, 5.5, 3.0, '15.12.2016 00:00'],
                                 [7.7, 2.1, 7.2, '15.12.2016 00:02']]
        result = prep.replace_missing_data_with_interpolations(data_with_time_stamps)
        result_values = [x[:-1] for x in result]
        result_time_stamps = [x[-1] for x in result]
        des_values = np.array([[1.2, 2.5, 7.6],
                               [6.1, 7.3, 8.5],
                               [5.1, 7.9, 6.2],
                               [4.1, 8.5, 3.9],
                               [3.1, 9.1, 1.6],
                               [2.9, 5.5, 3.0],
                               [5.3, 3.8, 5.1],
                               [7.7, 2.1, 7.2]])
        des_time_stamps = ['14.12.2016 23:55',
                           '14.12.2016 23:56',
                           '14.12.2016 23:57',
                           '14.12.2016 23:58',
                           '14.12.2016 23:59',
                           '15.12.2016 00:00',
                           '15.12.2016 00:01',
                           '15.12.2016 00:02']
        npt.assert_almost_equal(np.asarray(result_values), des_values)
        self.assertEqual(result_time_stamps, des_time_stamps)

        data_with_time_stamps = [[1.2, 7.6, '30.11.2016 23:57'],
                                 [6.1, 8.5, '30.11.2016 23:59'],
                                 [4.4, 1.2, '01.12.2016 00:01'],
                                 [0.0, 8.0, '01.12.2016 00:12'],
                                 [1.2, 4.4, '01.12.2016 00:16']]
        result = prep.replace_missing_data_with_interpolations(data_with_time_stamps)
        result_values = [x[:-1] for x in result]
        result_time_stamps = [x[-1] for x in result]
        des_values = np.array([[1.2, 7.6],
                               [3.65, 8.05],
                               [6.1, 8.5],
                               [5.25, 4.85],
                               [4.4, 1.2],
                               [0.0, 8.0],
                               [0.3, 7.1],
                               [0.6, 6.2],
                               [0.9, 5.3],
                               [1.2, 4.4]])
        des_time_stamps = ['30.11.2016 23:57',
                           '30.11.2016 23:58',
                           '30.11.2016 23:59',
                           '01.12.2016 00:00',
                           '01.12.2016 00:01',
                           '01.12.2016 00:12',
                           '01.12.2016 00:13',
                           '01.12.2016 00:14',
                           '01.12.2016 00:15',
                           '01.12.2016 00:16']
        npt.assert_almost_equal(np.asarray(result_values), des_values)
        self.assertEqual(result_time_stamps, des_time_stamps)

    def test_get_simulation_data_of_longest_state_sequence(self):
        s_a_pair_feature_order = ['charge_rate_in_W', 'generation_in_W', 'load_in_W']
        s_a_pairs = np.array([[1.2, 2.5, 7.6],
                              [2.8, 6.2, 4.4],
                              [4.4, 9.9, 1.2],
                              [3.5, 7.7, 2.1],
                              [2.6, 5.5, 3.0],
                              [7.7, 2.6, 7.2]])
        time_stamps = ['04.11.2016 11:56',
                       '04.11.2016 11:57',
                       '04.11.2016 11:58',
                       '04.11.2016 12:00',
                       '04.11.2016 12:01',
                       '04.11.2016 12:04']
        s_a_pairs, states, time_stamps = prep.get_simulation_data_of_longest_state_sequence(s_a_pairs, 
                                                                                            time_stamps,
                                                                                            s_a_pair_feature_order)
        des_s_a_pairs = np.array([[1.2, 2.5, 7.6],
                                  [2.8, 6.2, 4.4],
                                  [4.4, 9.9, 1.2]])
        des_states = np.array([[2.5, 7.6],
                               [6.2, 4.4],
                               [9.9, 1.2]])
        des_time_stamps = ['04.11.2016 11:56',
                           '04.11.2016 11:57',
                           '04.11.2016 11:58']
        des_test_states = np.array([[7.7, 2.1],
                                    [5.5, 3.0]])
        des_test_time_stamps = ['04.11.2016 12:00',
                                '04.11.2016 12:01']
        npt.assert_array_equal(des_s_a_pairs, s_a_pairs)
        npt.assert_array_equal(des_states, states)
        self.assertEqual(des_time_stamps, time_stamps)

        s_a_pair_feature_order = ['charge_rate_in_W', 'generation_in_W', 'load_in_W']
        s_a_pairs = np.array([[1.2, 2.5, 7.6],
                              [2.8, 6.2, 4.4],
                              [4.4, 9.9, 1.2],
                              [3.5, 7.7, 2.1],
                              [2.6, 5.5, 3.0],
                              [7.7, 2.6, 7.2]])
        time_stamps = ['30.11.2016 23:43',
                       '30.11.2016 23:44',
                       '30.11.2016 23:59',
                       '01.12.2016 00:00',
                       '01.12.2016 00:01',
                       '01.12.2016 00:02']
        s_a_pairs, states, time_stamps = prep.get_simulation_data_of_longest_state_sequence(s_a_pairs, 
                                                                                       time_stamps,
                                                                                       s_a_pair_feature_order)
        des_s_a_pairs = np.array([[4.4, 9.9, 1.2],
                                  [3.5, 7.7, 2.1],
                                  [2.6, 5.5, 3.0],
                                  [7.7, 2.6, 7.2]])
        des_states = np.array([[9.9, 1.2],
                               [7.7, 2.1],
                               [5.5, 3.0],
                               [2.6, 7.2]])
        des_time_stamps = ['30.11.2016 23:59',
                           '01.12.2016 00:00',
                           '01.12.2016 00:01',
                           '01.12.2016 00:02']
        des_test_states = np.array([[2.5, 7.6],
                                    [6.2, 4.4]])
        des_test_time_stamps = ['30.11.2016 23:43',
                                '30.11.2016 23:44']
        npt.assert_array_equal(des_s_a_pairs, s_a_pairs)
        npt.assert_array_equal(des_states, states)
        self.assertEqual(des_time_stamps, time_stamps)

        s_a_pair_feature_order = ['generation_in_W', 'charge_rate_in_W', 'load_in_W']
        s_a_pairs = np.array([[1.2, 2.5, 7.6],
                              [2.8, 6.2, 4.4],
                              [4.4, 9.9, 1.2],
                              [3.5, 7.7, 2.1],
                              [8.2, 2.8, 1.8],
                              [3.6, 0.9, 5.4],
                              [2.7, 6.8, 5.3],
                              [2.6, 5.5, 3.0],
                              [7.7, 2.6, 7.2]])
        time_stamps = ['31.12.2018 23:59',
                       '01.01.2019 00:00',
                       '01.01.2019 00:01',
                       '01.01.2019 00:03',
                       '01.01.2019 00:04',
                       '01.01.2019 01:05',
                       '01.01.2019 01:06',
                       '01.01.2019 01:07',
                       '01.01.2019 01:08']
        s_a_pairs, states, time_stamps = prep.get_simulation_data_of_longest_state_sequence(s_a_pairs, 
                                                                                       time_stamps,
                                                                                       s_a_pair_feature_order)
        des_s_a_pairs = np.array([[3.6, 0.9, 5.4],
                                  [2.7, 6.8, 5.3],
                                  [2.6, 5.5, 3.0],
                                  [7.7, 2.6, 7.2]])                                                                          
        des_states = np.array([[3.6, 5.4],
                               [2.7, 5.3],
                               [2.6, 3.0],
                               [7.7, 7.2]])
        des_time_stamps = ['01.01.2019 01:05',
                           '01.01.2019 01:06',
                           '01.01.2019 01:07',
                           '01.01.2019 01:08']
        des_test_states = np.array([[1.2, 7.6],
                                    [2.8, 4.4],
                                    [4.4, 1.2]])
        des_test_time_stamps = ['31.12.2018 23:59',
                                '01.01.2019 00:00',
                                '01.01.2019 00:01']
        npt.assert_array_equal(des_s_a_pairs, s_a_pairs)
        npt.assert_array_equal(des_states, states)
        self.assertEqual(des_time_stamps, time_stamps)

    def test_get_idxs_of_state_sequences(self):
        temporal_resolution_in_mins = 1
        time_stamps = ['04.11.2016 11:56',
                       '04.11.2016 11:57',
                       '04.11.2016 11:58',
                       '04.11.2016 12:00',
                       '04.11.2016 12:01',
                       '04.11.2016 12:04']
        des_result = [[0,1,2],[3,4]]
        result = prep.get_idxs_of_state_sequences(time_stamps, temporal_resolution_in_mins)
        self.assertEqual(result, des_result)

        temporal_resolution_in_mins = 5
        time_stamps = ['04.11.2016 23:36',
                       '04.11.2016 23:46',
                       '04.11.2016 23:56',
                       '05.11.2016 00:01',
                       '05.11.2016 00:06',
                       '05.11.2016 00:16']
        des_result = [[2,3,4]]
        result = prep.get_idxs_of_state_sequences(time_stamps, temporal_resolution_in_mins)
        self.assertEqual(result, des_result)

        temporal_resolution_in_mins = 10
        time_stamps = ['31.12.2016 23:53',
                       '01.01.2017 00:03',
                       '01.01.2017 00:13',
                       '02.01.2017 13:03',
                       '02.01.2017 13:13',
                       '02.01.2017 13:23']
        des_result = [[0,1,2],[3,4,5]]
        result = prep.get_idxs_of_state_sequences(time_stamps, temporal_resolution_in_mins)
        self.assertEqual(result, des_result)

    def test_subsample_data(self):
        data = np.array([[1.2, 2.5, 7.6],
                         [2.8, 6.2, 4.4],
                         [4.4, 9.9, 1.2],
                         [3.5, 7.7, 2.1],
                         [2.6, 5.5, 3.0],
                         [8.2, 6.2, 4.1],
                         [7.7, 2.6, 7.2]])
        time_stamps = ['04.11.2016 11:56',
                       '04.11.2016 11:57',
                       '04.11.2016 11:58',
                       '04.11.2016 11:59',
                       '04.11.2016 12:00',
                       '04.11.2016 12:01',
                       '04.11.2016 12:02']
        res_data, res_time_stamps = prep.subsample_data(data, time_stamps, temporal_resolution_in_mins=3)
        des_data = np.array([[1.2, 2.5, 7.6],
                             [3.5, 7.7, 2.1],
                             [7.7, 2.6, 7.2]])
        des_time_stamps = ['04.11.2016 11:56',
                           '04.11.2016 11:59',
                           '04.11.2016 12:02']
        npt.assert_array_equal(res_data, des_data)
        self.assertEqual(res_time_stamps, des_time_stamps)

        data = np.array([[1.2, 2.5, 7.6],
                         [2.8, 6.2, 4.4],
                         [4.4, 9.9, 1.2],
                         [3.5, 7.7, 2.1],
                         [8.2, 2.8, 1.8],
                         [3.6, 0.9, 5.4],
                         [2.7, 6.8, 5.3],
                         [2.6, 5.5, 3.0],
                         [7.7, 2.6, 7.2]])
        time_stamps = ['31.12.2018 23:59',
                       '01.01.2019 00:00',
                       '01.01.2019 00:01',
                       '01.01.2019 00:02',
                       '01.01.2019 00:03',
                       '01.01.2019 01:04',
                       '01.01.2019 01:05',
                       '01.01.2019 01:06',
                       '01.01.2019 01:07']
        res_data, res_time_stamps = prep.subsample_data(data, time_stamps, temporal_resolution_in_mins=5)
        des_data = np.array([[1.2, 2.5, 7.6],
                             [3.6, 0.9, 5.4]])
        des_time_stamps = ['31.12.2018 23:59',
                           '01.01.2019 01:04']
        npt.assert_array_equal(res_data, des_data)
        self.assertEqual(res_time_stamps, des_time_stamps)

        data = np.array([[1.2, 2.5, 7.6],
                         [2.8, 6.2, 4.4],
                         [4.4, 9.9, 1.2],
                         [3.5, 7.7, 2.1],
                         [8.2, 2.8, 1.8],
                         [3.6, 0.9, 5.4],
                         [6.1, 8.0, 2.9],
                         [2.7, 6.8, 5.3],
                         [2.6, 5.5, 3.0],
                         [1.8, 3.6, 3.7],
                         [7.7, 2.6, 7.2]])
        time_stamps = ['31.12.2018 23:59',
                       '01.01.2019 00:00',
                       '01.01.2019 00:01',
                       '01.01.2019 00:02',
                       '01.01.2019 00:03',
                       '01.01.2019 01:04',
                       '01.01.2019 01:05',
                       '01.01.2019 01:06',
                       '01.01.2019 01:07',
                       '01.01.2019 01:08',
                       '01.01.2019 01:09']
        res_data, res_time_stamps = prep.subsample_data(data, time_stamps, temporal_resolution_in_mins=5)
        des_data = np.array([[1.2, 2.5, 7.6],
                             [3.6, 0.9, 5.4],
                             [7.7, 2.6, 7.2]])
        des_time_stamps = ['31.12.2018 23:59',
                           '01.01.2019 01:04',
                           '01.01.2019 01:09']
        npt.assert_array_equal(res_data, des_data)
        self.assertEqual(res_time_stamps, des_time_stamps)

        data = np.array([[2.4, 6.2],
                         [9.9, 0.1],
                         [5.5, 4.0],
                         [2.6, 9.4]])
        time_stamps = ['28.11.2016 13:42',
                       '28.11.2016 13:43',
                       '28.11.2016 13:44',
                       '28.11.2016 13:45']
        res_data, res_time_stamps = prep.subsample_data(data, time_stamps, temporal_resolution_in_mins=1)
        npt.assert_array_equal(res_data, data)
        self.assertEqual(res_time_stamps, time_stamps)

if __name__ == "__main__":
    unittest.main()

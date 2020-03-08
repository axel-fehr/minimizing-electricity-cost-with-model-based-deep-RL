import unittest
import numpy as np
from numpy import testing as npt
from preprocessing import merging_data as md

class TestTest(unittest.TestCase):

    def setUp(self):
        self.dict1 = {}
        self.dict2 = {}

    def tearDown(self):
        pass

    def test_find_overlap_beginnings(self):
        time_stamp_values_1 = np.asarray(['04.10.2016 14:14',
                                          '04.10.2016 14:15'])
        time_stamp_values_2 = np.asarray(['04.10.2016 14:14',
                                          '04.10.2016 14:15'])
        self.dict1['time_stamps'] = time_stamp_values_1
        self.dict2['time_stamps'] = time_stamp_values_2
        result = md.find_overlap_beginnings(self.dict1, self.dict2)
        npt.assert_array_equal(result, np.array([[0, 0]]))

        time_stamp_values_1 = np.asarray(['04.10.2016 14:14',
                                          '04.10.2016 14:15'])
        time_stamp_values_2 = np.asarray(['04.10.2016 14:15',
                                          '04.10.2016 14:16'])
        self.dict1['time_stamps'] = time_stamp_values_1
        self.dict2['time_stamps'] = time_stamp_values_2
        result = md.find_overlap_beginnings(self.dict1, self.dict2)
        npt.assert_array_equal(result, np.array([[1, 0]]))

        time_stamp_values_1 = np.asarray(['04.10.2016 14:15',
                                          '04.10.2016 14:16'])
        time_stamp_values_2 = np.asarray(['04.10.2016 14:14',
                                          '04.10.2016 14:15'])
        self.dict1['time_stamps'] = time_stamp_values_1
        self.dict2['time_stamps'] = time_stamp_values_2
        result = md.find_overlap_beginnings(self.dict1, self.dict2)
        npt.assert_array_equal(result, np.array([[0, 1]]))

        time_stamp_values_1 = np.asarray(['15.12.2016 23:55',
                                          '15.12.2016 23:56',
                                          '15.12.2016 23:58',
                                          '15.12.2016 23:59',
                                          '16.12.2016 00:00',
                                          '16.12.2016 00:01'])
        time_stamp_values_2 = np.asarray(['15.12.2016 23:55',
                                          '15.12.2016 23:56',
                                          '15.12.2016 23:57',
                                          '15.12.2016 23:59', 
                                          '16.12.2016 00:00',
                                          '16.12.2016 00:01'])
        self.dict1['time_stamps'] = time_stamp_values_1
        self.dict2['time_stamps'] = time_stamp_values_2
        result = md.find_overlap_beginnings(self.dict1, self.dict2)
        npt.assert_array_equal(result, np.array([[0, 0], [3, 3]]))

        time_stamp_values_1 = np.asarray(['30.11.2016 23:57',
                                          '30.11.2016 23:59',
                                          '01.12.2016 00:00',
                                          '01.12.2016 00:01',
                                          '01.12.2016 00:03',
                                          '01.12.2016 00:04'])
        time_stamp_values_2 = np.asarray(['30.11.2016 23:57',
                                          '30.11.2016 23:58',
                                          '30.11.2016 23:59',
                                          '01.12.2016 00:00',
                                          '01.12.2016 00:01',
                                          '01.12.2016 00:02',
                                          '01.12.2016 00:03',
                                          '01.12.2016 00:04'])
        self.dict1['time_stamps'] = time_stamp_values_1
        self.dict2['time_stamps'] = time_stamp_values_2
        result = md.find_overlap_beginnings(self.dict1, self.dict2)
        npt.assert_array_equal(result, np.array([[0, 0], [1, 2], [4,6]]))

        time_stamp_values_1 = np.asarray(['31.12.2018 23:57',
                                          '31.12.2018 23:58',
                                          '31.12.2018 23:59',
                                          '01.01.2019 00:00',
                                          '01.01.2019 00:01',
                                          '01.01.2019 00:02',
                                          '01.01.2019 00:03',
                                          '01.01.2019 00:04'])
        time_stamp_values_2 = np.asarray(['31.12.2018 23:57',
                                          '31.12.2018 23:59',
                                          '01.01.2019 00:00',
                                          '01.01.2019 00:01',
                                          '01.01.2019 00:03',
                                          '01.01.2019 00:04'])
        self.dict1['time_stamps'] = time_stamp_values_1
        self.dict2['time_stamps'] = time_stamp_values_2
        result = md.find_overlap_beginnings(self.dict1, self.dict2)
        npt.assert_array_equal(result, np.array([[0,0], [2, 1], [6,4]]))

        time_stamp_values_1 = np.asarray(['04.10.2016 14:14',
                                          '04.10.2016 14:15'])
        time_stamp_values_2 = np.asarray(['05.10.2016 14:14',
                                          '05.10.2016 14:15'])
        self.dict1['time_stamps'] = time_stamp_values_1
        self.dict2['time_stamps'] = time_stamp_values_2
        self.assertRaises(ValueError, md.find_overlap_beginnings, self.dict1, self.dict2)

    def test_find_overlap_ends(self):
        time_stamp_values_1 = np.asarray(['04.10.2016 14:14',
                                          '04.10.2016 14:15'])
        time_stamp_values_2 = np.asarray(['04.10.2016 14:14',
                                          '04.10.2016 14:15'])
        self.dict1['time_stamps'] = time_stamp_values_1
        self.dict2['time_stamps'] = time_stamp_values_2
        result = md.find_overlap_ends(self.dict1, self.dict2, np.array([[0, 0]]))
        npt.assert_array_equal(result, np.array([[1, 1]]))

        time_stamp_values_1 = np.asarray(['04.10.2016 14:14',
                                          '04.10.2016 14:15'])
        time_stamp_values_2 = np.asarray(['04.10.2016 14:15',
                                          '04.10.2016 14:16'])
        self.dict1['time_stamps'] = time_stamp_values_1
        self.dict2['time_stamps'] = time_stamp_values_2
        result = md.find_overlap_ends(self.dict1, self.dict2, np.array([[1, 0]]))
        npt.assert_array_equal(result, np.array([[1, 0]]))

        time_stamp_values_1 = np.asarray(['04.10.2016 14:15',
                                          '04.10.2016 14:16'])
        time_stamp_values_2 = np.asarray(['04.10.2016 14:14',
                                          '04.10.2016 14:15'])
        self.dict1['time_stamps'] = time_stamp_values_1
        self.dict2['time_stamps'] = time_stamp_values_2
        result = md.find_overlap_ends(self.dict1, self.dict2, np.array([[0, 1]]))
        npt.assert_array_equal(result, np.array([[0, 1]]))

        time_stamp_values_1 = np.asarray(['15.12.2016 23:55',
                                          '15.12.2016 23:56',
                                          '15.12.2016 23:58',
                                          '15.12.2016 23:59',
                                          '16.12.2016 00:00',
                                          '16.12.2016 00:01'])
        time_stamp_values_2 = np.asarray(['15.12.2016 23:55',
                                          '15.12.2016 23:56',
                                          '15.12.2016 23:57',
                                          '15.12.2016 23:59', 
                                          '16.12.2016 00:00',
                                          '16.12.2016 00:01'])
        self.dict1['time_stamps'] = time_stamp_values_1
        self.dict2['time_stamps'] = time_stamp_values_2
        result = md.find_overlap_ends(self.dict1, self.dict2, np.array([[0, 0], [3, 3]]))
        npt.assert_array_equal(result, np.array([[1, 1], [5, 5]]))

        time_stamp_values_1 = np.asarray(['30.11.2016 23:57',
                                          '30.11.2016 23:59',
                                          '01.12.2016 00:00',
                                          '01.12.2016 00:01',
                                          '01.12.2016 00:03',
                                          '01.12.2016 00:04'])
        time_stamp_values_2 = np.asarray(['30.11.2016 23:57',
                                          '30.11.2016 23:58',
                                          '30.11.2016 23:59',
                                          '01.12.2016 00:00',
                                          '01.12.2016 00:01',
                                          '01.12.2016 00:02',
                                          '01.12.2016 00:03',
                                          '01.12.2016 00:04'])
        self.dict1['time_stamps'] = time_stamp_values_1
        self.dict2['time_stamps'] = time_stamp_values_2
        result = md.find_overlap_ends(self.dict1, self.dict2, np.array([[0, 0], [1, 2], [4,6]]))
        npt.assert_array_equal(result, np.array([[0,0],[3,4],[5,7]]))

        time_stamp_values_1 = np.asarray(['31.12.2018 23:57',
                                          '31.12.2018 23:58',
                                          '31.12.2018 23:59',
                                          '01.01.2019 00:00',
                                          '01.01.2019 00:01',
                                          '01.01.2019 00:02',
                                          '01.01.2019 00:03',
                                          '01.01.2019 00:04'])
        time_stamp_values_2 = np.asarray(['31.12.2018 23:57',
                                          '31.12.2018 23:59',
                                          '01.01.2019 00:00',
                                          '01.01.2019 00:01',
                                          '01.01.2019 00:03',
                                          '01.01.2019 00:04'])
        self.dict1['time_stamps'] = time_stamp_values_1
        self.dict2['time_stamps'] = time_stamp_values_2
        result = md.find_overlap_ends(self.dict1, self.dict2, np.array([[0,0], [2, 1], [6,4]]))
        npt.assert_array_equal(result, np.array([[0,0],[4,3],[7,5]]))

    def test_merge_idxs_into_single_array(self):
        arr1 = np.array([[1, 2]])
        arr2 = np.array([[5, 6]])
        result = md.merge_idxs_into_single_array(arr1, arr2)
        npt.assert_array_equal(result, np.array([[[1,5],[2,6]]]))

        arr1 = np.array([[1, 2],[3, 4]])
        arr2 = np.array([[5, 6],[7, 8]])
        result = md.merge_idxs_into_single_array(arr1, arr2)
        npt.assert_array_equal(result, np.array([[[1,5],[2,6]],[[3,7],[4,8]]]))

    def test_get_start_and_end_indices_of_overlaps(self):
        time_stamp_values_1 = np.asarray(['04.10.2016 14:14',
                                          '04.10.2016 14:15'])
        time_stamp_values_2 = np.asarray(['04.10.2016 14:14',
                                          '04.10.2016 14:15'])
        self.dict1['time_stamps'] = time_stamp_values_1
        self.dict2['time_stamps'] = time_stamp_values_2
        result = md.get_start_and_end_indices_of_overlaps(self.dict1, self.dict2)
        npt.assert_array_equal(result, np.array([[[0,1],[0,1]]]))

        time_stamp_values_1 = np.asarray(['12.11.2016 05:34',
                                          '12.11.2016 05:35'])
        time_stamp_values_2 = np.asarray(['12.11.2016 05:35',
                                          '12.11.2016 05:36'])
        self.dict1['time_stamps'] = time_stamp_values_1
        self.dict2['time_stamps'] = time_stamp_values_2
        result = md.get_start_and_end_indices_of_overlaps(self.dict1, self.dict2)
        npt.assert_array_equal(result, np.array([[[1,1],[0,0]]]))

        time_stamp_values_1 = np.asarray(['04.10.2016 14:15',
                                          '04.10.2016 14:16'])
        time_stamp_values_2 = np.asarray(['04.10.2016 14:14',
                                          '04.10.2016 14:15'])
        self.dict1['time_stamps'] = time_stamp_values_1
        self.dict2['time_stamps'] = time_stamp_values_2
        result = md.get_start_and_end_indices_of_overlaps(self.dict1, self.dict2)
        npt.assert_array_equal(result, np.array([[[0,0],[1,1]]]))

        time_stamp_values_1 = np.asarray(['15.12.2016 23:55',
                                          '15.12.2016 23:56',
                                          '15.12.2016 23:58',
                                          '15.12.2016 23:59',
                                          '16.12.2016 00:00',
                                          '16.12.2016 00:01'])
        time_stamp_values_2 = np.asarray(['15.12.2016 23:55',
                                          '15.12.2016 23:56',
                                          '15.12.2016 23:57',
                                          '15.12.2016 23:59', 
                                          '16.12.2016 00:00',
                                          '16.12.2016 00:01'])
        self.dict1['time_stamps'] = time_stamp_values_1
        self.dict2['time_stamps'] = time_stamp_values_2
        result = md.get_start_and_end_indices_of_overlaps(self.dict1, self.dict2)
        npt.assert_array_equal(result, np.array([[[0,1],[0,1]],[[3,5],[3,5]]]))

        time_stamp_values_1 = np.asarray(['30.11.2016 23:57',
                                          '30.11.2016 23:59',
                                          '01.12.2016 00:00',
                                          '01.12.2016 00:01',
                                          '01.12.2016 00:03',
                                          '01.12.2016 00:04'])
        time_stamp_values_2 = np.asarray(['30.11.2016 23:57',
                                          '30.11.2016 23:58',
                                          '30.11.2016 23:59',
                                          '01.12.2016 00:00',
                                          '01.12.2016 00:01',
                                          '01.12.2016 00:02',
                                          '01.12.2016 00:03',
                                          '01.12.2016 00:04'])
        self.dict1['time_stamps'] = time_stamp_values_1
        self.dict2['time_stamps'] = time_stamp_values_2
        result = md.get_start_and_end_indices_of_overlaps(self.dict1, self.dict2)
        npt.assert_array_equal(result, np.array([[[0,0],[0,0]],[[1,3],[2,4]],[[4,5],[6,7]]]))

        time_stamp_values_1 = np.asarray(['31.12.2018 23:57',
                                          '31.12.2018 23:58',
                                          '31.12.2018 23:59',
                                          '01.01.2019 00:00',
                                          '01.01.2019 00:01',
                                          '01.01.2019 00:02',
                                          '01.01.2019 00:03',
                                          '01.01.2019 00:04'])
        time_stamp_values_2 = np.asarray(['31.12.2018 23:57',
                                          '31.12.2018 23:59',
                                          '01.01.2019 00:00',
                                          '01.01.2019 00:01',
                                          '01.01.2019 00:03',
                                          '01.01.2019 00:04'])
        self.dict1['time_stamps'] = time_stamp_values_1
        self.dict2['time_stamps'] = time_stamp_values_2
        result = md.get_start_and_end_indices_of_overlaps(self.dict1, self.dict2)
        npt.assert_array_equal(result, np.array([[[0,0],[0,0]],[[2,4],[1,3]],[[6,7],[4,5]]]))

        time_stamp_values_1 = np.asarray(['04.10.2016 14:14',
                                          '04.10.2016 14:15'])
        time_stamp_values_2 = np.asarray(['05.10.2016 14:14',
                                          '05.10.2016 14:15'])
        self.dict1['time_stamps'] = time_stamp_values_1
        self.dict2['time_stamps'] = time_stamp_values_2
        self.assertRaises(ValueError, md.get_start_and_end_indices_of_overlaps, self.dict1, self.dict2)

    def test_remove_idxs_of_single_element_overlaps(self):
        overlap_idxs = np.array([[[0,1],[0,1]]])
        result = md.remove_idxs_of_single_element_overlaps(overlap_idxs)
        npt.assert_array_equal(result, np.array([[[0,1],[0,1]]]))

        overlap_idxs = np.array([[[1,1],[0,0]]])
        self.assertRaises(ValueError, md.remove_idxs_of_single_element_overlaps, overlap_idxs)

        overlap_idxs = np.array([[[0,0],[1,1]]])
        self.assertRaises(ValueError, md.remove_idxs_of_single_element_overlaps, overlap_idxs)

        overlap_idxs = np.array([[[0,1],[0,1]], [[3,4],[4,5]]])
        result = md.remove_idxs_of_single_element_overlaps(overlap_idxs)
        npt.assert_array_equal(result, np.array([[[0,1],[0,1]],[[3,4],[4,5]]]))

        overlap_idxs = np.array([[[0,1],[0,1]], [[3,3],[4,4]], [[6,8],[8,10]], [[9,9],[12,12]]])
        result = md.remove_idxs_of_single_element_overlaps(overlap_idxs)
        npt.assert_array_equal(result, np.array([[[0,1],[0,1]], [[6,8],[8,10]]]))

    def test_merge_energy_system_and_weather_data(self):
        overlap_idxs = np.array([[[0,1],[0,1]]])
        self.dict1['arbitrary_values_1'] = np.array([1,2])
        self.dict1['time_stamps'] = np.array(["01.01.2016 02:45", "01.01.2016 02:46"])
        self.dict2['arbitrary_values_2'] = np.array([4,5])
        self.dict2['time_stamps'] = np.array(["01.01.2016 02:45", "01.01.2016 02:46"])
        data, keys = md.merge_energy_system_and_weather_data(self.dict1, self.dict2, overlap_idxs)
        self.assertEqual(data, [[1,4,166,"01.01.2016 02:45"], [2,5,167,"01.01.2016 02:46"]])
        self.assertEqual(keys, ['arbitrary_values_1', 'arbitrary_values_2', 'time_stamps'])

        overlap_idxs = np.array([[[0,1],[0,1]],[[3,4],[4,5]]])
        self.dict1['arbitrary_values_1'] = np.array([1,2,3,4,5,6,7,8,9])
        self.dict1['time_stamps'] = np.array(["01.01.2016 02:45", "01.01.2016 02:46", "01.01.2016 02:47",
                                              "01.01.2016 02:48", "01.01.2016 02:49"])
        self.dict2['arbitrary_values_2'] = np.array([10,11,12,13,14,15,16,17,18,19])
        self.dict2['time_stamps'] = np.array(["01.01.2016 02:45", "01.01.2016 02:46", "01.01.2016 02:47",
                                              "01.01.2016 02:48", "01.01.2016 02:48", "01.01.2016 02:49"])
        data, keys = md.merge_energy_system_and_weather_data(self.dict1, self.dict2, overlap_idxs)
        self.assertEqual(data, [[1,10,166,"01.01.2016 02:45"], [2,11,167,"01.01.2016 02:46"],
                                [4,14,169,"01.01.2016 02:48"], [5,15,170,"01.01.2016 02:49"]])
        self.assertEqual(keys, ['arbitrary_values_1', 'arbitrary_values_2', 'time_stamps'])

        overlap_idxs = np.array([[[0,2],[1,3]],[[3,4],[4,5]]])
        self.dict1['arbitrary_values_1'] = np.array([1,2,3,4,5,6,7,8,9])
        self.dict1['time_stamps'] = np.array(["15.11.2016 23:55", "15.11.2016 23:56", "15.11.2016 23:57",
                                              "15.11.2016 23:59", "16.11.2016 00:00", "16.11.2016 00:01"])
        self.dict2['arbitrary_values_2'] = np.array([10,11,12,13,14,15,16,17,18,19])
        self.dict2['time_stamps'] = np.array(["15.11.2016 23:54", "15.11.2016 23:55", "15.11.2016 23:56",
                                              "15.11.2016 23:57", "15.11.2016 23:59", "16.11.2016 00:00",
                                              "16.11.2016 00:02"])
        data, keys = md.merge_energy_system_and_weather_data(self.dict1, self.dict2, overlap_idxs)
        self.assertEqual(data, [[1,11,1436,"15.11.2016 23:55"], [2,12,1437,"15.11.2016 23:56"], 
                                [3,13,1438,"15.11.2016 23:57"], [4,14,1440,"15.11.2016 23:59"],
                                [5,15,1,"16.11.2016 00:00"]])

        overlap_idxs = np.array([[[0,1],[0,1]]])
        self.dict1['arbitrary_values_1'] = np.array([1,2,3])
        self.dict1['time_stamps'] = np.array(["01.01.2016 02:45", "01.01.2016 02:46"])
        self.dict2['arbitrary_values_2'] = np.array([4,5,6])
        self.dict2['time_stamps'] = np.array(["01.01.2016 02:46", "01.01.2016 02:47"])
        self.assertRaises(ValueError, md.merge_energy_system_and_weather_data, self.dict1, self.dict2, overlap_idxs)

    def test_time_stamp_string_to_integer(self):
        result = md.time_stamp_string_to_integer("01.01.2016 02:45")
        self.assertEqual(result, 166)

        result = md.time_stamp_string_to_integer("11.11.2016 15:29")
        self.assertEqual(result, 930)

        result = md.time_stamp_string_to_integer("23.12.2016 00:00")
        self.assertEqual(result, 1)

        result = md.time_stamp_string_to_integer("30.10.2017 23:59")
        self.assertEqual(result, 1440)

        self.assertRaises(ValueError, md.time_stamp_string_to_integer, "01.01.2016 45:23")
        self.assertRaises(ValueError, md.time_stamp_string_to_integer, "01.01.2016 12:61")

    def are_subsequent_time_stamps(self):
        self.assertTrue(md.are_subsequent_time_stamps("01.01.2016 02:45", "11.11.2016 02:46"))
        self.assertTrue(md.are_subsequent_time_stamps("13.12.2017 17:19", "13.12.2017 17:20"))
        self.assertTrue(md.are_subsequent_time_stamps("22.01.2019 23:59", "23.01.2019 00:00"))
        self.assertTrue(md.are_subsequent_time_stamps("30.11.2016 23:59", "01.12.2016 00:00"))
        self.assertTrue(md.are_subsequent_time_stamps("31.12.2018 23:59", "01.01.2019 00:00"))
        self.assertFalse(md.are_subsequent_time_stamps("01.01.2016 02:45", "11.11.2016 02:47"))
        self.assertFalse(md.are_subsequent_time_stamps("01.01.2016 02:45", "11.11.2016 03:46"))
        self.assertFalse(md.are_subsequent_time_stamps("18.11.2016 14:19", "26.12.2016 18:02"))
        self.assertFalse(md.are_subsequent_time_stamps("22.01.2019 23:59", "23.01.2019 00:01"))
        self.assertFalse(md.are_subsequent_time_stamps("30.11.2016 23:59", "01.12.2016 00:01"))
        self.assertFalse(md.are_subsequent_time_stamps("31.12.2018 23:59", "01.01.2019 00:01"))
        self.assertFalse(md.are_subsequent_time_stamps("22.01.2019 23:59", "24.01.2019 00:00"))
        self.assertFalse(md.are_subsequent_time_stamps("30.11.2016 23:59", "02.12.2016 00:00"))
        self.assertFalse(md.are_subsequent_time_stamps("31.12.2018 23:59", "02.01.2019 00:00"))

if __name__ == "__main__":
    unittest.main()
-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: mariadb
-- Generation Time: May 08, 2024 at 11:38 AM
-- Server version: 10.4.8-MariaDB-1:10.4.8+maria~bionic
-- PHP Version: 8.2.8

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `theraplot`
--

-- --------------------------------------------------------

--
-- Table structure for table `datasets`
--

CREATE TABLE `datasets` (
  `dataset_id` varchar(225) NOT NULL,
  `project_name` varchar(100) NOT NULL,
  `dataset_link` varchar(200) NOT NULL,
  `uploaded_date` datetime NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `datasets_runs`
--

CREATE TABLE `datasets_runs` (
  `run_id` varchar(225) NOT NULL,
  `rmse` varchar(20) NOT NULL,
  `mae` varchar(20) NOT NULL,
  `r_square` varchar(20) NOT NULL,
  `epoch` varchar(20) NOT NULL,
  `runtime` varchar(200) NOT NULL,
  `run_csv_path` varchar(200) NOT NULL,
  `num_rows` varchar(50) NOT NULL DEFAULT '0',
  `dataset_id` varchar(225) NOT NULL,
  `run_date` datetime NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `user_id` varchar(225) NOT NULL,
  `passcode` varchar(300) NOT NULL,
  `user_name` varchar(200) NOT NULL,
  `date` datetime NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`user_id`, `passcode`, `user_name`, `date`) VALUES
('qbaGuV4mN1qz', '$2b$12$Q1ToYjLfqcgC1TGkhzbngenbpi5B5YvZF4nXgT/s1yim76pOeq236', 'theresa', '2024-03-14 07:05:43');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `datasets`
--
ALTER TABLE `datasets`
  ADD PRIMARY KEY (`dataset_id`);

--
-- Indexes for table `datasets_runs`
--
ALTER TABLE `datasets_runs`
  ADD PRIMARY KEY (`run_id`),
  ADD KEY `dataset_id` (`dataset_id`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`user_id`);

--
-- Constraints for dumped tables
--

--
-- Constraints for table `datasets_runs`
--
ALTER TABLE `datasets_runs`
  ADD CONSTRAINT `datasets_runs_ibfk_1` FOREIGN KEY (`dataset_id`) REFERENCES `datasets` (`dataset_id`) ON DELETE CASCADE ON UPDATE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;

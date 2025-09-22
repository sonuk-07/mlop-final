CREATE DATABASE IF NOT EXISTS shoppers_db;

CREATE USER IF NOT EXISTS 'sonu'@'%' IDENTIFIED BY 'Yunachan10';
GRANT ALL PRIVILEGES ON shoppers_db.* TO 'sonu'@'%';

CREATE USER IF NOT EXISTS 'sonu'@'localhost' IDENTIFIED BY 'Yunachan10';
GRANT ALL PRIVILEGES ON shoppers_db.* TO 'sonu'@'localhost';

FLUSH PRIVILEGES;
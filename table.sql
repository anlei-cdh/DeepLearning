CREATE TABLE `dl_classify_data` (
  `hash_id` BIGINT(20) NOT NULL,
  `name` VARCHAR(32) NOT NULL,
  `type` VARCHAR(32) NOT NULL,
  `score` FLOAT(11) NOT NULL,
  PRIMARY KEY (`hash_id`)
) ENGINE=INNODB DEFAULT CHARSET=utf8;

CREATE TABLE `dl_detection_data` (
  `id` INT(11) NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(32) NOT NULL,
  `top` FLOAT(11) NOT NULL,
  `left` FLOAT(11) NOT NULL,
  `width` FLOAT(11) NOT NULL,
  `height` FLOAT(11) NOT NULL,
  `type` VARCHAR(32) NOT NULL,
  `score` FLOAT(32) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=INNODB DEFAULT CHARSET=utf8;

CREATE TABLE `dl_poem_data` (
  `id` BIGINT(20) NOT NULL,
  `content` VARCHAR(512) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=INNODB DEFAULT CHARSET=utf8;

INSERT INTO dl_poem_data(`id`,`content`) VALUES(1,'飞伐三千里，随明水一曾。不觉心间剡，人看江上春。舍今三十家，林下九比边。九白通千里，潇始依旧山。');
INSERT INTO dl_poem_data(`id`,`content`) VALUES(2,'飞与便恋去，依旧翠青钟。雨涉微鸟起，僧多岸不枯。马于规涛谓，沙虎箕零群。应有门前化，堪闻逐遇开。');
INSERT INTO dl_poem_data(`id`,`content`) VALUES(3,'');
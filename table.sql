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

CREATE TABLE `dl_poetize_data` (
  `id` INT(11) NOT NULL,
  `content` VARCHAR(512) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=INNODB DEFAULT CHARSET=utf8;

INSERT INTO dl_poetize_data(`id`,`content`) VALUES(1,'天芋仙幸住，停经穴黠魂。野稀和必上，出水寺生山。闻舍见朝帝，昨夜是且忧。树栅当寒减，亲风只融称。');
INSERT INTO dl_poetize_data(`id`,`content`) VALUES(2,'落流雁去外，春风枝树新。冥心凝开每，绕乐世钓心。叶拙容归弓，春风水上闲。槛来何处宿，行步威入瀛。');
INSERT INTO dl_poetize_data(`id`,`content`) VALUES(3,'飞毛惆仙处，晚更还分明。薛远春经位，春生积夜应。处多中更老，遥带岸莺尘。孤彩法以雨，星桥及有泉。');
UPDATE `dl_poetize_data` SET content = '' WHERE id = 3;
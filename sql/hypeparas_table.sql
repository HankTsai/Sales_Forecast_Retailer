use WebFormPT_RTM
go
create table SFI_F16_WEEKLY_HYPEPARAM_OPT(
	STOREID				varchar(10)		not null,
	START_DATE			date			not null,
	END_DATE			date			not null,
	MODEL_SCORE			decimal(15,3)	not null,
	ETA					decimal(10,3)	not null,
	GUMMA				int				not null,
	MAX_DEPTH			int				not null,
	SUBSAMPEL			decimal(10,3)	not null,
	REG_LAMBDA			int				not null,
	REG_ALPHA			int				not null,
	N_ESTIMATORS		int				not null,
	MIN_CHILD_WEIGHT	int				not null,
	COLSAMPLE_BYTREE	decimal(10,3)	not null,
	COLSAMPLE_BYLEVEL	decimal(10,3)	not null,
	RECORD_TIME			date			not null
);

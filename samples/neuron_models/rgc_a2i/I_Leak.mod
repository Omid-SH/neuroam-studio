TITLE The leakage current channel (IL)
:
: Added three more ionic channehls to the modehl from Kameneva 2011 to distinguish the difference between ON and OFF RGCs. 
: Written by Javad Paknahad July 5, 2018

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX IL
	NONSPECIFIC_CURRENT iL
	RANGE eL, gL
}

UNITS {
	(molar) = (1/liter)
	(mM) = (millimolar)
	(mA) = (milliamp)
	(mV) = (millivolt)
}


PARAMETER {
	gL = 0.001 (mho/cm2)
	eL = -60        (mV)
	dt              (ms)
	v               (mV)
}

ASSIGNED {
	iL (mA/cm2)
}

BREAKPOINT { iL = gL*(v - eL) }

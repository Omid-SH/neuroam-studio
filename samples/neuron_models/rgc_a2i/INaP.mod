TITLE The persistent Na+ current (INaP)
:
: Added three more ionic channehls to the modehl from Kameneva 2011 to distinguish the difference between ON and OFF RGCs. 
: Written by Javad Paknahad July 5, 2018

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX INaP
	NONSPECIFIC_CURRENT inaP
	RANGE gnaPbar, enaP
	RANGE p_inf
	RANGE tau_p
	RANGE p_exp
}

UNITS {
	(molar) = (1/liter)
	(mM) = (millimolar)
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {

	: For three added channels
	gnaPbar	= 0.012 (mho/cm2)
	enaP = 35        (mV)
	dt              (ms)
	v               (mV)
}

STATE {
	p
}

INITIAL {
: The initial values were determined at a resting value of -66.3232 mV in a single-compartment
: at -60 mV
	:	p = 5.8137e-04
: at -65 mV
		p = 2.4321e-04
}

ASSIGNED {
	inaP  (mA/cm2)
	p_inf 
	tau_p 
	p_exp 
	a_p b_p 
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	inaP = gnaPbar * p * (v - enaP)
}


DERIVATIVE states {
	evaluate_fct(v)
	p' = -(a_p + b_p)* p + a_p
}

:UNITSOFF

PROCEDURE evaluate_fct(v(mV)) { LOCAL a,b
	
: NaP channel 

	if (v < -40) {
		a_p = (0.025+0.14*exp((v+40)/10))/(1+exp(-1*(v+48)/10))
		b_p = (1-((1+exp(-1*(v+48)/10))^(-1)))/(0.025+0.14*exp((v+40)/10))
		tau_p = 1 / (a_p + b_p)
		p_inf = a_p * tau_p
	} else {
	
		a_p = (0.02+0.145*exp(-1*(v+40)/10))/(1+exp(-1*(v+48)/10))
		b_p = (1-((1+exp(-1*(v+48)/10))^(-1)))/(0.02+0.145*exp((v+40)/10))
		tau_p = 1 / (a_p + b_p)
		p_inf = a_p * tau_p
	}

}
UNITSON

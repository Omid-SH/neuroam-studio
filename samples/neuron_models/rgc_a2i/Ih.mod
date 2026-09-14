TITLE The hyperpolarization-activated (Ih)
:
: Added three more ionic channehls to the modehl from Kameneva 2011 to distinguish the difference between ON and OFF RGCs. 
: Written by Javad Paknahad July 5, 2018

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX Ih
	NONSPECIFIC_CURRENT ih
	RANGE ghbar, eh
	RANGE l_inf
	RANGE tau_l
	RANGE l_exp
}

UNITS {
	(molar) = (1/liter)
	(mM) = (millimolar)
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {

	: For three added channels
	ghbar	= 0.012 (mho/cm2)
	eh = 0     (mV)
	dt              (ms)
	v               (mV)
	a0t=0.011      	(/ms)
	celsius 		(degC)
	q10=4.5
	qtl=1
	kl=-8

}

STATE {
	l
}

INITIAL {
: The initial values were determined at a resting value of -66.3232 mV in a single-compartment
: at -60 mV
		l=0.6788
}

ASSIGNED {
	ih  (mA/cm2)
	l_inf
	tau_l
	l_exp
	a_l b_l
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	ih= ghbar * l * (v-eh)
}

DERIVATIVE states {
	evaluate_fct(v)
	l' =  (l_inf - l)/tau_l
}

:UNITSOFF

FUNCTION alpt(v(mV)) {
  		alpt = exp(0.08316*(v + 75))
}

FUNCTION bett(v(mV)) {
  		bett = exp(0.033264*(v + 75))
}

PROCEDURE evaluate_fct(v (mV)) {
        LOCAL a,qt
        qt=q10^((celsius-33)/10)
        a = alpt(v)
        l_inf = 1/(1 + exp(-(v + 81)/kl))
        tau_l = bett(v)/(qtl*qt*a0t*(1+a))
}

UNITSON

from sympy import E, S, I, im as Im_func, re as Re_func, log, Float
import mpmath

CUSTOM_NAMESPACE = {
    'catalan': S.Catalan,
    'Catalan': S.Catalan,
    'A': Float(mpmath.glaisher),
    'ln': log,
    'im': Im_func,
    'Im': Im_func,
    're': Re_func,
    'Re': Re_func,
    'I': I,
    'e': E,
}

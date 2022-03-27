from math import fmod,pi

def normalize_angle(angle):
    a = fmod(fmod(angle, 2.0*pi) + 2.0*pi, 2.0*pi)
    if a > pi:
        a -= 2.0 *pi
    return a

import numpy as np

def Gr(g,beta,DT,L,nu):
    return (g*beta*DT*L**3)/(nu**2)

def Nu_vertical_plate_local(Gr_x,Pr):
    if Pr<0:
        print('Pr is too low.')
        return 'warning'
    gPr=(0.75*Pr**.5)/(0.609+1.221*Pr**.5+1.238*Pr)**.25
    return((Gr_x/4)**.25*gPr)

def Nu_vertical_plate_avg(Gr,Pr):
    if Pr<0:
        print('Pr is too low.')
        return 'warning'
    gPr=(0.75*Pr**.5)/(0.609+1.221*Pr**.5+1.238*Pr)**.25
    return((4/3)*(Gr/4)**.25*gPr)


#WILL DAVIE - Computing Ex 3
#––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Aims:
# To numerically evaluate the Fresnel Diffraction Intergral.

# Section A: one dimensional diffraction 
# Section B: Exploration of changing fresnel number
# Section C: two dimensional diffraction

import numpy as np
from math import *
import matplotlib.pyplot as plt
import scipy.integrate as si


#––––––––––––––––––––––––––––––––––––––––––––––––––––––

#SECTION A

#––––––––––––––––––––––––––––––––––––––––––––––––––––––

#Init parameters:

wavelength = 1 * 10**(-6)
wavenumber = (2*pi) / wavelength
screendistance = 0.02
E_0 = 0.01
epsilon = 8.854 *10**(-12)
c_light = 3 * 10**8

N_screen = 500

#Approximate width of the central maximum
def central_max_width(wavelength,z,width):
    MW = z * wavelength / width
    return MW

#Fresnel number is a very useful number to quantify observed diffraction patterns.

def fresnel_number(slit_width,wavelength,distoscreen):
    F = slit_width**2 / (wavelength*distoscreen)
    return F


#–––––––––––––––––––––––––––––––––––––––––––––––––––––
class oneDiffraction():

#One dimensional diffraction.

    def __init__(self,wavenumber,distoscreen,x_scr):
        #Class useful to store these parameters
        self.k = wavenumber
        self.z = distoscreen
        self.x_scr = x_scr

    #Equation 4 has three intergrals:
    # - LHS: f(x) , RHS1: g(x), and RHS2: h(x). 

    def f_func(self,x_ap):
        #LHS intergral
        self.f = np.exp(  ( (complex(1j) * self.k) /  (2*self.z) ) * (self.x_scr - x_ap)**2 )
        return self.f
        
    def g_func(self,x_ap):
        #RHS cosine intergral
        self.g = np.cos( ( (self.k) /  (2*self.z) ) * (self.x_scr - x_ap)**2 ) 
        return self.g


    def h_func(self,x_ap):
        #RHS sine intergral
        self.h = np.sin( ( (self.k) /  (2*self.z) ) * (self.x_scr - x_ap)**2 )
        return self.h

#–––––––––––––––––––––––––––––––––––––––––––––––––––––

#SECTION A RUN FUNCTION

def section_a(x_1_dash,x_2_dash,screendistance,N):

    width_a = x_2_dash - x_1_dash

    maximum_width = central_max_width(wavelength,screendistance,width_a)

    fresnel_num = fresnel_number(width_a,wavelength,screendistance)

    #Here i found it useful to plot the size of the screen in relation to the central peak width. 
    #It works well for lower fresnel numbers but eventually does not encapsulate the whole pattern. 
    if fresnel_num < 1:
        x_screen_range = np.linspace(-5*maximum_width,5*maximum_width,N_screen)
    else:
        x_screen_range = np.linspace(5*x_1_dash,5*x_2_dash,N_screen)


    x_aperture_range = np.linspace(x_1_dash,x_2_dash,N)

    #Intensity array evaluated by LH part of Eq 4
    LHS_Intensity = []

    #Intensity array evaluated by RH part of Eq 4
    RHS_Intensity = [] 

    for x in x_screen_range:

        dif_object = oneDiffraction(wavenumber,screendistance,x)

        LHS_inner = []
        RHS_Real_inner = []


        #FINDING LHS INTENSITY - Simpson Method

        for x_dash in x_aperture_range:
            #complex exponential:
            LHS_inner.append(dif_object.f_func(x_dash))
            
        LHS_intergral = si.simps(LHS_inner,x_aperture_range)
        LHS_E = ( (wavenumber*E_0) / (2*pi*screendistance) ) * LHS_intergral
        LHS_I = epsilon * c_light * np.abs(  LHS_E * np.conjugate(LHS_E) )
        LHS_Intensity.append(LHS_I)

        #FINDING RHS INTENSITY - .quad()

        RHS_Real,r_err = si.quad(dif_object.g_func,x_1_dash,x_2_dash)
        RHS_Imag,i_err = si.quad(dif_object.h_func,x_1_dash,x_2_dash)
        RHS_E = ( (wavenumber*E_0) / (2*pi*screendistance) ) * (complex(RHS_Real,RHS_Imag))
        #Equation 3
        RHS_I =  epsilon * c_light * np.abs(  RHS_E * np.conjugate(RHS_E) )
        RHS_Intensity.append(RHS_I)

    return x_screen_range,LHS_Intensity,RHS_Intensity


#END OF SECTION
#–––––––––––––––––––––––––––––––––––––––––––––––––––––

#SECTION B - This is simply using the section A function and will be shown later.

#–––––––––––––––––––––––––––––––––––––––––––––––––––––

#SECTION C

#––––––––––––––––––––––––––––––––––––––––––––––––––––––

#This section is very similar to section A

#One dimensional diffraction.

class twoDiffraction():

    def __init__(self,wavenumber,z,y_scr,x_scr):
        #Class useful to store these parameters
        self.k = wavenumber
        self.z = z
        self.x_scr = x_scr
        self.y_scr = y_scr

    #Equation 5 can be re-written and represented by real cosine and imaginary sine parts

    def fresnel_real(self,y_ap,x_ap):
        #LHS intergral
        f = np.cos( ( (self.k) /  (2*self.z) ) * ((self.x_scr - x_ap)**2 + (self.y_scr - y_ap)**2) )
        return f
         
    def fresnel_imag(self,y_ap,x_ap):
        #RHS cosine intergral
        f = np.sin( ( (self.k) /  (2*self.z) ) * ((self.x_scr - x_ap)**2 + (self.y_scr - y_ap)**2) )
        return f


def section_c(y_1_dash,y_2_dash,x_1_dash,x_2_dash,distoscreen_fres,N):


    width_x = x_2_dash - x_1_dash
    width_y = y_2_dash - y_1_dash

    maximum_width_x = central_max_width(wavelength,distoscreen_fres,width_x)
    maximum_width_y = central_max_width(wavelength,distoscreen_fres,width_y)

    fresnel_num = fresnel_number(width_x,wavelength,screendistance)
    print(fresnel_num)

    if fresnel_num < 1:
        x_screen_range = np.linspace(-3*maximum_width_x,3*maximum_width_x,N)
        y_screen_range = np.linspace(-3*maximum_width_y,3*maximum_width_y,N)

    else:
        x_screen_range = np.linspace(1.5*x_1_dash,1.5*x_2_dash,N) 
        y_screen_range = np.linspace(1.5*y_1_dash,1.5*y_2_dash,N)

    intensity = np.zeros( (N,N) )

    #This time we must loop over both dimentions
    for y in y_screen_range:
        for x in x_screen_range:
            dif_object = twoDiffraction(wavenumber,distoscreen_fres,y,x)

            real,r_err = si.dblquad(dif_object.fresnel_real,y_1_dash,y_2_dash,x_1_dash,x_2_dash)
            imag,i_err = si.dblquad(dif_object.fresnel_imag,y_1_dash,y_2_dash,x_1_dash,x_2_dash)

            E = ( (wavenumber*E_0) / (2*pi*distoscreen_fres) ) * (complex(real,imag))
            I =  epsilon * c_light * np.abs(  E * np.conjugate(E) )
            intensity[list(x_screen_range).index(x),list(y_screen_range).index(y)] = I

        print(list(y_screen_range).index(y)/N * 100,'%')
        

    return intensity

#plot = section_c(-4*10**-5,4*10**-5,-4*10**-5,4*10**-5,0.02,200)
#plt.imshow(plot)
#plt.show()


#END OF SECTION
#–––––––––––––––––––––––––––––––––––––––––––––––––––––

#SECTION D

#––––––––––––––––––––––––––––––––––––––––––––––––––––––

#Using essentially the same code as section C:

def section_d_cicular(y_1_dash,y_2_dash,distoscreen_fres,N):
    
    radius = (y_2_dash-y_1_dash) / 2

    maximum_width_r = central_max_width(wavelength,distoscreen_fres,radius*2)

    fresnel_num = fresnel_number(radius*2,wavelength,screendistance)

    if fresnel_num < 1:
        x_screen_range = np.linspace(-3*maximum_width_r,3*maximum_width_r,N)
        y_screen_range = np.linspace(-3*maximum_width_r,3*maximum_width_r,N)

    else:
        x_screen_range = np.linspace(1.5*y_1_dash,1.5*y_2_dash,N) 
        y_screen_range = np.linspace(1.5*y_1_dash,1.5*y_2_dash,N)

    def x1(y1):
        x = -np.sqrt(radius ** 2 - y1 ** 2)
        return x
    def x2(y2):
        x = np.sqrt(radius ** 2 - y2 ** 2)
        return x


    intensity = np.zeros( (N,N) )

    for y in y_screen_range:
        for x in x_screen_range:
            dif_object = twoDiffraction(wavenumber,distoscreen_fres,y,x)

            real,r_err = si.dblquad(dif_object.fresnel_real,y_1_dash,y_2_dash,x1,x2)
            imag,i_err = si.dblquad(dif_object.fresnel_imag,y_1_dash,y_2_dash,x1,x2)

            E = ( (wavenumber*E_0) / (2*pi*distoscreen_fres) ) * (complex(real,imag))
            I =  epsilon * c_light * np.abs(  E * np.conjugate(E) )
            intensity[list(x_screen_range).index(x),list(y_screen_range).index(y)] = I
        print(list(y_screen_range).index(y)/N * 100,'%')
    return intensity


#plot = section_d_cicular(-1*10**-5,1*10**-5,0.02,50)
#plt.imshow(plot)
#plt.show()


#END OF SECTION
#–––––––––––––––––––––––––––––––––––––––––––––––––––––

#EXTENSION: Comparing Fresnel and Fraunhofer diffraction

#––––––––––––––––––––––––––––––––––––––––––––––––––––––

#Fraunhofer model of diffraction Intensity is described by the square of a sinc function

#For 1d:

def fraun_I(x_screen_range,x_1_dash,x_2_dash,wavenumber,distoscreen,I_max,N):

    #width of apeture
    ap_width  = x_2_dash - x_1_dash

    I = []

    for x in x_screen_range:
        if x == 0:
            intensity = I_max
        else:
            intensity = I_max * (np.sin( (wavenumber*ap_width*x)/( 2* distoscreen ) ))**2 / ( (wavenumber*ap_width*x)/( 2* distoscreen ) )**2

        I.append(intensity)

    return I


def extension(x_1_dash,x_2_dash,screendistance,N):

    xvals,LHS_Intensity,RHS_Intensity = section_a(x_1_dash,x_2_dash,screendistance,N)
    I_max = max(LHS_Intensity)

    fraun_intensity = fraun_I(xvals,x_1_dash,x_2_dash,wavenumber,screendistance,I_max,N_screen)


    print(sum((np.array(fraun_intensity)-np.array(RHS_Intensity))**2 / np.array(RHS_Intensity)) )

    fig,ax = plt.subplots()

    ax.plot(xvals,RHS_Intensity,label='Fresnel')
    ax.plot(xvals,fraun_intensity,label='Fraunhofer')
    ax.set_ylabel('Relative Intensity',fontsize=14)
    ax.set_xlabel('Screen co-ord (m)',fontsize=14)
    ax.set_title("Fresnel vs Fraunhofer",fontsize=18)
    ax.legend()
    plt.show()

#END OF SECTION
#––––––––––––––––––––––––––––––––––––––––––––––––––––––

#Commented Code used for report. Doesn't make much sense just random testing.



#def chi_squard(O,E):
 #   Y = (np.array(O)-np.array(E))**2 / np.array(E)
  #  X = np.sum(Y)
   # return X

#Comparing different values of N for Simpson method

#x_range_1,simps_1,quad_1 = section_a(-3*10**(-4),3*10**(-4),0.02,101)
#x_range_3,simps_3,quad_3 = section_a(-1*10**(-5),1*10**(-5),0.02,11)
##x_range_2,simps_2,quad_2 = section_a(-1*10**(-5),1*10**(-5),0.02,11)


#plot = section_c(-2*10**-4,2*10**-4,-1*10**-4,1*10**-4,0.02,100)
#plot = section_d_cicular(-1*10**-4,1*10**-4,0.005,100)

#fig_cs,ax_cs = plt.subplots()
#ax_cs.set_ylabel('Relative Intensity',fontsize=14)
#ax_cs.set_xlabel('Screen co-ord (m)',fontsize=14)
#ax_cs.set_title("Simpson's vs Quadrature at N = 101",fontsize=18)

#ax_cs.plot(x_range_3,simps_3,label='N=5')
#ax_cs.plot(x_range_2,simps_2,label='N=11')
#ax_cs.plot(x_range_1,simps_1,label='N=101')
#ax_cs.plot(x_range_1,simps_1,label='Simpson')
#ax_cs.plot(x_range_1,quad_1,label='Quadrature')
#ax_cs.legend()
#plt.show()

#fig_cq,ax_cq = plt.subplots()
#ax_cq.set_title("2D diffraction",fontsize=18)
#ax_cq.imshow(plot)
#plt.show()

#for i in np.linspace(0.5,20,5):
 #   extension(-i*10**(-5),i*10**(-5),0.02,301)



#––––––––––––––––––––––––––––––––––––––––––––––––––––––

#RUNNING - using Ex1 template

#––––––––––––––––––––––––––––––––––––––––––––––––––––––

MyInput = '0'
while MyInput != 'q':
    MyInput = input('Enter a choice, "a", "b", "c", "d", "e" or "q" to quit: ')
    print('You entered the choice: ',MyInput)

    if MyInput == 'a':
        print('You have chosen part (a)')
        print('––––––––––––––––––––––––––––––––––––––––––––––––––––––')
        print('Intensity plots using the complex exponential intergral and the cosine-sine intergrals with given parameters')
        print('––––––––––––––––––––––––––––––––––––––––––––––––––––––')

        #A plot using given parameters

        x_range,complexexp,sine = section_a(-1*10**(-5),1*10**(-5),0.02,101)
        fig_a, (ax_a_e,ax_a_s) = plt.subplots(2,figsize=(10,10))
        fig_a.suptitle('One dimensional diffraction using given parameters')
        ax_a_e.set(title='Simpson Method',xlabel='Screen co-ord (m)',ylabel='Relative Intensity')
        ax_a_e.plot(x_range,complexexp)
        ax_a_s.set(title='Quadrature Method',xlabel='Screen co-ord (m)',ylabel='Relative Intensity')
        ax_a_s.plot(x_range,sine)
        plt.show()


    elif MyInput == 'b' or MyInput == 'e':

        if MyInput == 'b':
            print('You have chosen part (b)')
            print('––––––––––––––––––––––––––––––––––––––––––––––––––––––')
            print('Changing section A parameters')
            print('––––––––––––––––––––––––––––––––––––––––––––––––––––––')
            


        if MyInput == 'e':
            print('You have chosen part (e)')
            print('––––––––––––––––––––––––––––––––––––––––––––––––––––––')
            print('Comparing the Fresnel and Fraunhofer model')
            print('––––––––––––––––––––––––––––––––––––––––––––––––––––––')

        #Choice between methods
        
        N_value = 0

        if MyInput == 'b':
            method_chosen = False

            #The option to choose the method of integration
            while method_chosen == False:
                method_input = input('Would you like to use the simpson or quadrature method of integration (S/q): ')
                if method_input == 'S':
                    method = 'simpson'
                    method_chosen = True
                else:
                    method = 'quadrature'
                    method_chosen = True

            #Choice of Num Points

            N_value = 0
            if method == 'simpson':
                while N_value == 0:
                    N_input = input('Enter a value of N (odd) to be used for the simpson method: ')
                    try:
                        N_value = int(N_input)
                    except:
                        print('Not a valid answer')
                        N_value = 0
            else:
                N_value = 101
        else:
            N_value = 101
            method = 'quadrature'
            
        #Choice of slit_width

        slit_width = 0

        while slit_width == 0:
            a_input = input('Enter a slit width (in units of 10^-5 m): ')
            try:
                slit_width = float(a_input) * 10**(-5)
                x2 = slit_width/2
                x1 = -x2
            except:
                print('this is not a valid answer')
                slit_width = 0

            print(x1,x2)

        #Choice of distance

        z = 0

        while z == 0:
            z_input = input('Enter a distance to the screen (in mm): ')
            try:
                z = float(z_input) * 10**(-3)
            except:
                print('this is not a valid answer')
                z = 0
        
        fresnel_num = fresnel_number(slit_width,wavelength,z)

        xvals,simp,quad = section_a(x1,x2,z,N_value)

        if MyInput == 'b':
            fig_b,ax_b = plt.subplots()

            if method == 'simpson':
                ax_b.set(title=f'Simpson method, N = {N_value} , Fresnel Number = {"{:.2f}".format(fresnel_num)}',xlabel='Screen co-ord (m)',ylabel='Relative Intensity')
                ax_b.plot(xvals,simp)
                plt.show()

            if method == 'quadrature':
                ax_b.set(title=f'Quadrature method, Fresnel Number = {"{:.2f}".format(fresnel_num)}',xlabel='Screen co-ord (m)',ylabel='Relative Intensity')
                ax_b.plot(xvals,quad)
                plt.show()
        
        if MyInput == 'e':
            extension(x1,x2,z,N_value)

 
        
    elif MyInput == 'c' or MyInput == 'd':

        #This uses a square apeture however for the report section_c() function can be used to plot a rectangular pattern. 
        if MyInput == 'c':
            print('You have chosen part (c)')
            print('––––––––––––––––––––––––––––––––––––––––––––––––––––––')
            print('Plotting a 2d diffraction pattern for a square apeture')
            print('––––––––––––––––––––––––––––––––––––––––––––––––––––––')
        if MyInput == 'd':
            print('You have chosen part (d)')
            print('––––––––––––––––––––––––––––––––––––––––––––––––––––––')
            print('Plotting a 2d diffraction pattern for a circular apeture')
            print('––––––––––––––––––––––––––––––––––––––––––––––––––––––')


        #Choice of Num Points

        N_c = 0
        while N_c == 0:
            N_input_c = input('Enter a value of N for the number of points on the screen (i.e NxN): ')
            try:
                N_c = int(N_input_c)
            except:
                print('Not a valid answer')
                N_c = 0
            
        #Choice of slit_width

        slit_width_c = 0

        while slit_width_c == 0:
            c_input = input('Enter a slit width(square) or diameter(circlular) (in units of 10^-5 m): ')
            try:
                slit_width_c = float(c_input) * 10**(-5)
                x2_c = slit_width_c/2
                x1_c = -x2_c
            except:
                print('this is not a valid answer')
                slit_width_c = 0

        #Choice of distance

        z_c = 0

        while z_c == 0:
            z_input_c = input('Enter a distance to the screen (in mm): ')
            try:
                z_c = float(z_input_c) * 10**(-3)
            except:
                print('this is not a valid answer')
                z_c = 0

        fresnel_num_c = fresnel_number(slit_width_c,wavelength,z_c)
        
        if MyInput == 'c':
            fig_c,ax_c = plt.subplots()
            ax_c.set(title=f'Square Apeture 2d, Fresnel Number = {"{:.2f}".format(fresnel_num_c)}')
            plot = section_c(x1_c,x2_c,x1_c,x2_c,z_c,N_c)
            ax_c.imshow(plot)
            plt.show()

        if MyInput == 'd':
            fig_d,ax_d = plt.subplots()
            ax_d.set(title=f'Circular Apeture 2d, Fresnel Number = {"{:.2f}".format(fresnel_num_c)}')
            plot = section_d_cicular(x1_c,x2_c,z_c,N_c)
            ax_d.imshow(plot)
            plt.show()


        

    elif MyInput != 'q':
        print('This is not a valid choice')

print('You have chosen to finish - goodbye.')
        

    






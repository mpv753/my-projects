from tkinter.ttk import Notebook
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import math
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.optimize import curve_fit
import warnings
from scipy.optimize import OptimizeWarning
from typing import Iterable
from tkinter import messagebox
from sympy import sympify, symbols
from scipy.optimize import fsolve
import os 


class Sea_Level_Anomaly_calculator(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Sea Level Anomaly Calculator")
        self.geometry("1024x768")
        self.minsize(400,200)
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (Main_menu, New_project, ECFC,LSC):
            frame = F(self.container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(Main_menu)
        
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

class Main_menu(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        button = tk.Button(self, text="New Project",command=lambda: controller.show_frame(New_project))
        button.pack()

        button1 = tk.Button(self, text="Exit",command=lambda: controller.destroy())
        button1.pack()
        

class New_project(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        backbutton = ttk.Button(self, text="Back", command=lambda: controller.show_frame(Main_menu))
        backbutton.place(relx=0.05, rely=0.025)
        self.radiob = tk.StringVar()
        self.radiob.set("False")
        self.fpath = tk.StringVar()
        self.minlat = tk.StringVar()
        self.minlat.set(-90)
        self.minlon = tk.StringVar()
        self.minlon.set(-180)
        self.maxlat = tk.StringVar()
        self.maxlat.set(90)
        self.maxlon = tk.StringVar()
        self.maxlon.set(180)

        label = tk.Label(self, text="Options", font=(None, 15))
        label.pack(pady=20, padx=20)

        file_button = tk.Button(self, text="Select your data", command=self.file_explorer)
        file_button.pack()
        self.path_label = tk.Label(self, font=(None, 10))
        self.path_label.pack(pady=20, padx=20)

        checkframe = tk.Frame(self, relief=tk.SUNKEN, bd=5)
        checkframe.pack(padx=10,pady=10)

        checklabel = tk.Label(checkframe, text="Do you want to preprocess your data?", font=(None, 10))
        checklabel.grid(column=0, row=0, columnspan=2, padx=10,pady=10)

        preprocess_rbutton = ttk.Radiobutton(checkframe, text="Yes", value="option1", variable=self.radiob)
        preprocess_rbutton.grid(padx=10,pady=10, row=1, column=0)

        preprocess_rbutton2 = ttk.Radiobutton(checkframe, text="No", value="option2", variable=self.radiob)
        preprocess_rbutton2.grid(padx=10,pady=10, row=1, column=1)

        pre_button = ttk.Button(checkframe, text="Calculate", command=self.selection)
        pre_button.grid(padx=10,pady=10, row=2, column=0, columnspan=2)

        maskframe = tk.Frame(self, relief=tk.SUNKEN, bd=5)
        maskframe.pack(padx=10,pady=10)

        firstlabel = ttk.Label(maskframe, text="Select area")
        firstlabel.grid(padx=10,pady=10, row=0, column=0, columnspan=4)

        latlabel = ttk.Label(maskframe, text="Latitude")
        latlabel.grid(padx=10,pady=10, row=1, column=1)

        lonlabel = ttk.Label(maskframe, text="Longitude")
        lonlabel.grid(padx=10,pady=10, row=1, column=2)

        maxlabel = ttk.Label(maskframe, text="max :")
        maxlabel.grid(padx=10,pady=10, row=2, column=0)

        minlabel = ttk.Label(maskframe, text="min :")
        minlabel.grid(padx=10,pady=10, row=3, column=0)

        latmin = ttk.Entry(maskframe, textvariable=self.minlat)
        latmin.grid(padx=10,pady=10, row=3, column=1)

        lonmin = ttk.Entry(maskframe, textvariable=self.minlon)
        lonmin.grid(padx=10,pady=10, row=3, column=2)

        lonmax = ttk.Entry(maskframe, textvariable=self.maxlon)
        lonmax.grid(padx=10,pady=10, row=2, column=2)

        latmax = ttk.Entry(maskframe, textvariable=self.maxlat)
        latmax.grid(padx=10,pady=10, row=2, column=1)

        maskbutton = ttk.Button(maskframe, text="Confirm", command=self.minmask)
        maskbutton.grid(padx=10,pady=10, row=4, column=0, columnspan=4, sticky='nesw')

        nextbutton = ttk.Button(self, text="Next", command=lambda: controller.show_frame(ECFC))
        nextbutton.pack()

    def file_explorer(self):
        file_path = filedialog.askopenfilename(initialdir="~/Desktop", title="Select your data")
        self.fpath.set(file_path)
        self.path_label.config(text=file_path)


    def read_data(self):
        data = []
        with open(self.fpath.get(), "r") as file:
            for line in file:
                data.append([float(x) for x in line.strip().split(",")])
        data_form = np.array(data)
        
        return data_form


    def selection(self):
        global meansla
        global oslaaccuracy
        meansla = None
        data = self.read_data()
        mean_third_column = np.mean(data[:, 2])
        meansla = mean_third_column
    
        if self.radiob.get() == "option1":
            
        
        # Subtract the mean from each element in the 3rd column
            data[:, 2] -= mean_third_column  
        
        # Calculate threshold for outlier removal
            threshold = np.sqrt(np.mean(data[:, 2]**2)) * 3
        
        # Filter out outliers based on the threshold
            indexes = np.logical_and(data[:, 2] <= threshold, data[:, 2] >= -threshold)
            self.clean_data = data[indexes]
            
        else:
            self.clean_data = self.read_data()
        
    # Extract oslaaccuracy if available
        if self.clean_data.shape[1] == 4:
            oslaaccuracy = self.clean_data[:, -1]
        else:
            oslaaccuracy = np.zeros_like(self.clean_data[:, 1])

        
    def minmask(self):
        lat = [float(self.minlat.get()), float(self.maxlat.get())]
        lon = [float(self.minlon.get()), float(self.maxlon.get())]

        lat_condition = np.logical_and(self.clean_data[:, 0] >= lat[0], self.clean_data[:, 0] <= lat[1])
        lon_condition = np.logical_and(self.clean_data[:, 1] >= lon[0], self.clean_data[:, 1] <= lon[1])

        indexes = np.logical_and(lat_condition, lon_condition)

        self.clean_data2 = self.clean_data[indexes]

        latitudes = np.radians(self.clean_data2[:, 0])
        longitudes = np.radians(self.clean_data2[:, 1])

        dlat = latitudes[:, np.newaxis] - latitudes
        dlon = longitudes[:, np.newaxis] - longitudes

        a = np.sin(dlat / 2) ** 2 + np.cos(latitudes[:, np.newaxis]) * np.cos(latitudes) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        radius_of_earth = 6371
        distances_matrix = radius_of_earth * c

        measurements = self.clean_data2[:, 2]
     
        products_matrix = np.outer(measurements, measurements)

        dmin=np.round(np.min(distances_matrix[distances_matrix!=0]))
        dmax=np.round(np.max(distances_matrix))
        recomended=round(2*np.mean(np.diag(distances_matrix,k=1)))
        minmax=f"The range of distances is: {dmin}-{dmax} Km,\n recomended step >= {recomended} Km "
        self.controller.frames[ECFC].range_dist.set(minmax)


        return products_matrix, distances_matrix,latitudes,longitudes,measurements


class ECFC(tk.Frame):
    def distance_classes(self,distances,products):
        step=float(self.controller.frames[ECFC].strscale.get())
        if step<=0:
            raise ValueError("Distance step must be greater than zero")
            
        step1=0
        step2=step
        class_dist=[0]
        class_cov=[np.mean(np.diag(products))]
        number_of_classes=round(np.max(distances)/step)
        count_class=[]
        for i in range(number_of_classes):
            mask=np.where(np.logical_and(distances>step1,distances<=step2))
            count_class.append(len(mask[0]))
            if i==number_of_classes-1:
                count_class.append(len(mask[0]))

            class_dist.append(step2)
            
            class_cov.append(np.mean(products[mask]))
            step2+=step
            step1+=step            
        global xpoint
        global ypoint
        xpoint=class_dist
        ypoint=class_cov
        return class_dist,class_cov,count_class

    def plot_data(self):
        # Clear previous plots
        self.clear_plots()

        # Call distance_clusters_separation to get data
        prods,data,lat,lon,measurements = self.controller.frames[New_project].minmask()
     
        

        midpoints, rms_values,count= self.distance_classes(data, prods)

        # Plot midpoints vs RMS values

        self.plot_midpoints_rms(midpoints, rms_values)
        
        
        # Plot histogram of the bins
        self.plot_histogram(midpoints,count)
        self.geoscatter(lat,lon,measurements)
     
    def clear_plots(self):
    # Clear plots from the "Midpoints vs RMS" tab
        for widget in self.tab_midpoints_rms.winfo_children():
            widget.destroy()

    # Clear plots from the "Histogram of Bins" tab
        for widget in self.tab_histogram.winfo_children():
            widget.destroy()

   # Clear plots from the "Sea level anomalies plot" tab 
        for widget in self.SLA.winfo_children():
            widget.destroy()

    def plot_midpoints_rms(self, midpoints, rms_values):
        ind=(~np.isnan(rms_values))
        mid=np.array(midpoints)[ind]
        r=np.array(rms_values)[ind]
        r*=10000
     
    # Clear previous plot
        plt.clf()
    
    # Create a new figure
    
        fig = plt.Figure(figsize=(8,8), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_xlim(left=np.min(mid)-1,right=1.05*np.max(mid))
        ax.scatter(mid, r, marker='o', color='blue')  # Plot points
        ax.plot(mid, r, color='red', linestyle='-', linewidth=2, label='Line Plot')
        ax.set_title('Empirical Covariance Function')
        ax.set_xlabel('Distance (Km)')
        ax.set_ylabel('Covariance (cm$^2$)')

    # Embed the plot in the Canvas within the tab
        canvas = FigureCanvasTkAgg(fig, master=self.tab_midpoints_rms)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def plot_histogram(self, midpoints,count):
        # Clear previous plot
        plt.clf()

        # Create a new figure
        fig = plt.Figure(figsize=(8,8), dpi=100)
        ax = fig.add_subplot(111)
        # ax.hist(midpoints,count, color='blue', alpha=0.7, rwidth=0.85)
        ax.set_title('Distance classes of pairs')
        ax.set_xlabel('Distance (Km)')
        ax.set_ylabel('Frequency')

        bin_widths = [midpoints[i+1] - midpoints[i] for i in range(len(midpoints) - 1)]
        bin_widths.append(0)  # Append 0 for the last bin
        ax.bar(midpoints, count, width=bin_widths, color='blue', alpha=0.7)
        

        # Embed the plot in the Canvas within the tab
        canvas = FigureCanvasTkAgg(fig, master=self.tab_histogram)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def geoscatter(self, latitudes, longitudes, values, **kwargs):
        latitudes_deg = np.degrees(latitudes)
        longitudes_deg = np.degrees(longitudes)
    

        values+=meansla

    
    # Create a figure for plotting
        fig = plt.figure(figsize=(8,8))

    # Define the projection (Plate Carrée in this case)
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    
    # Adjust subplot layout to minimize whitespace
        plt.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1)

    # Add coastlines, land, and other features
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.set_extent([np.min(longitudes_deg) - 1, np.max(longitudes_deg) + 1,
                   np.min(latitudes_deg) - 1, np.max(latitudes_deg) + 1],
                  crs=ccrs.PlateCarree())
    
    # Convert data to map projection coordinates
        x, y = ax.projection.transform_points(ccrs.PlateCarree(), longitudes_deg, latitudes_deg)[:, :2].T
        

        # Add gridlines with coordinates
        gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False  # Remove labels from the top side
        gl.right_labels = False  # Show labels on the right side
        gl.bottom_labels = True  # Show labels on the bottom side
        gl.left_labels = True # Remove labels from the left side
    
    # Plot data points on the map
        if values is not None:
            scatter = ax.scatter(x, y, c=values*100, s=10, transform=ccrs.PlateCarree(), **kwargs)
            plt.colorbar(scatter, label='Sea Level Anomaly values (cm)', cmap="warm", shrink=0.5)
        else:
            ax.scatter(x, y, transform=ccrs.PlateCarree(), **kwargs)

    # Use FigureCanvasTkAgg to embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.SLA)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
 

    def exponential_decay(self, x, a, b, c):
        return a * np.exp(-b * x) + c

    def exponential_decay_2(self, x, a, b, c, d):
        return a * np.exp(-b * x) + c * np.exp(-d * x)

    def exponential_decay_3(self, x, a, b, c, d, e):
        return a * np.exp(-b * x) + c * np.exp(-d * x) + e * np.exp(-x)
    
    def damped_oscillation(self, x, a, b, c, d):
        return a * np.exp(-b * x) * np.cos(c * x) + d

    def damped_oscillation_2(self, x, a, b, c, d, e):
        return a * np.exp(-b * x) * np.cos(c * x) + d * np.exp(-e * x)

    def damped_oscillation_3(self, x, a, b, c, d, e, f):
        return a * np.exp(-b * x) * np.cos(c * x) + d * np.exp(-e * x) + f * np.exp(-x)

    def gaussian_decay(self, x, a, b, c):
        return a * np.exp(-(x - b)**2 / (2 * c**2))

    def gaussian_decay_2(self, x, a, b, c, d):
        return a * np.exp(-(x - b)**2 / (2 * c**2)) + d

    def gaussian_decay_3(self, x, a, b, c, d, e):
        return a * np.exp(-(x - b)**2 / (2 * c**2)) + d * np.exp(-e * x)
# Define polynomial functions
    def linear_function(self,x, a, b):
        return a * x + b

    def quadratic_function(self,x, a, b, c):
        return a * x**2 + b * x + c

    def cubic_function(self,x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x + d

    def quartic_function(self,x, a, b, c, d, e):
        return a * x**4 + b * x**3 + c * x**2 + d * x + e

    def quintic_function(self,x, a, b, c, d, e, f):
        return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f

    def sextic_function(self,x, a, b, c, d, e, f, g):
        return a * x**6 + b * x**5 + c * x**4 + d * x**3 + e * x**2 + f * x + g

    def septic_function(self,x, a, b, c, d, e, f, g, h):
        return a * x**7 + b * x**6 + c * x**5 + d * x**4 + e * x**3 + f * x**2 + g * x + h

    def octic_function(self,x, a, b, c, d, e, f, g, h, i):
        return a * x**8 + b * x**7 + c * x**6 + d * x**5 + e * x**4 + f * x**3 + g * x**2 + h * x + i
    

    def find_x_given_y(self,f, params, y):
    # Define a function that calculates the difference between f(x) and y
        def equation_to_solve(x):
            return f(x, *params) - y

    # Use fsolve to find the root of the equation (where f(x) - y = 0)
        x_solution = fsolve(equation_to_solve, x0=0)  # Initial guess for x

        return x_solution



    
    def analytical_models(self):
        # Define a dictionary to categorize functions
        fit_functions = {
             'Exponential': {
                'a*e^(-b*x) + c': self.exponential_decay,
                'a*e^(-b*x) + c*e^(-d*x)': self.exponential_decay_2,
                'a*e^(-b*x) + c*e^(-d*x) + e*e^(-x)': self.exponential_decay_3
            },
            'Damped Oscillation': {
                'a*e^(-b*x)*cos(c*x) + d': self.damped_oscillation,
                'a*e^(-b*x)*cos(c*x) + d*e^(-e*x)': self.damped_oscillation_2,
                'a*e^(-b*x)*cos(c*x) + d*e^(-e*x) + f*e^(-x)': self.damped_oscillation_3
            },
            'Gaussian Decay': {
                'a*e^(-(x - b)^2 / (2*c^2))': self.gaussian_decay,
                'a*e^(-(x - b)^2 / (2*c^2)) + d': self.gaussian_decay_2,
                'a*e^(-(x - b)^2 / (2*c^2)) + d*e^(-e*x)': self.gaussian_decay_3
            },
    'Polynomial': {
        'a*x+b': self.linear_function,
        'a*x^2+b*x+c': self.quadratic_function,
        'a*x^3+b*x^2+c*x+d': self.cubic_function,
        'a*x^4+b*x^3+c*x^2+d*x+e': self.quartic_function,
        'a*x^5+b*x^4+c*x^3+d*x^2+e*x+f': self.quintic_function,
        'a*x^6+b*x^5+c*x^4+d*x^3+e*x^2+f*x+g': self.sextic_function,
        'a*x^7+b*x^6+c*x^5+d*x^4+e*x^3+f*x^2+g*x+h': self.septic_function,
        'a*x^8+b*x^7+c*x^6+d*x^5+e*x^4+f*x^3+g*x^2+h*x+i': self.octic_function
    
}}

        selected_function=self.functionvar.get()
       
       

        
            
    # Iterate over the functions in each category
        for category, functions in fit_functions.items():
            for func_string, func in functions.items():
                if selected_function == func_string:
                    selected_f = func  # Assign the selected function
                  
                    break  # Exit the loop once the match is found
                
        
        x=xpoint
        y=ypoint
        
        first_run=self.controller.frames[ECFC].first_run.get()
        if first_run==True:
            for i in range(len(y)):
                y[i]=y[i]*10000
                self.controller.frames[ECFC].first_run.set(False)
        try:
        # Perform curve fitting
            params= curve_fit(selected_f, x, y)
          
            return selected_f,params[0]
        except (OptimizeWarning, RuntimeError):
        # Display a message box if curve fitting fails
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            messagebox.showerror("Curve Fitting Failed", "Curve fitting failed. Try a different function or adjust the data.")

    def plot_analytical_model(self):
        model, params= self.controller.frames[ECFC].analytical_models()
        

        plt.clf()
        for widget in self.analytical_model.winfo_children():
            widget.destroy()    

        x_values = np.linspace(min(xpoint), max(xpoint), 10000)  # Adjust the range as needed
       
        y_values=np.ones(len(x_values))
   
        for i in range(len(x_values)):
            y_values[i]=model(x_values[i],*params)
        
        global Cor_length_value
        Cor_length_value=self.controller.frames[ECFC].find_x_given_y(model,params,y_values[0]/2)
        
        
        #Cor_length=Cor_len
       

        y=ypoint

        
        fig=plt.figure(figsize=(8,8))
        ax=fig.add_subplot(111)
        ECF=ax.plot(xpoint,y,c="b",label="Empirical Covariance Function")
        ECFC2=ax.scatter(xpoint,y,c="b")
        Anal_mod=ax.plot(x_values,y_values,c="r",label="Analytical model: \n"+self.functionvar.get())
        ax.set_xlabel('Distance (Km)')
        ax.set_ylabel('Covariance (cm$^2$)')
        ax.set_title('Analytical Model')
        ax.legend(loc="best")
        
        ax.text(0.5,0.2,f"Correlation length: {round(Cor_length_value[0])} Km\n Covariance: {round(y[0],1)} cm$^2$")

        
      
        

        canvas = FigureCanvasTkAgg(fig, master=self.analytical_model)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.first_run=True
        self.notebook = Notebook(self)
        self.notebook.grid(row=0, column=2,rowspan=10)
        self.notebook.grid_propagate(False)
        self.notebook2=Notebook(self)
        self.notebook2.grid(row=1, column=0)
        # Create tabs for each plot
        self.tab_midpoints_rms = tk.Frame(self.notebook)
        self.tab_histogram = tk.Frame(self.notebook)
        self.SLA= tk.Frame(self.notebook)
        self.analytical_model=tk.Frame(self.notebook)
        self.calc_options=tk.Frame(self.notebook2)
        self.calc_options.grid(row=1, column=1, sticky="n")  # Anchor to the top of the notebook
        # self.calc_options.grid_propagate(False)  # Prevent resizing
        self.backframe=tk.Frame(self)
        self.backframe.grid(row=0,column=0)

        self.first_run=tk.BooleanVar()
        self.first_run.set(True)
       

        # Add tabs to the notebook
        
        self.notebook.add(self.tab_midpoints_rms, text='Empirical Covariance Function')
        self.notebook.add(self.tab_histogram, text='Distance Classes')
        self.notebook.add(self.SLA, text='Sea Level Anomalies')
        self.notebook.add(self.analytical_model,text="Empirical Covariance Function Analytical Model")
        self.notebook2.add(self.calc_options)

        backbutton = ttk.Button(self.backframe, text="Back", command=lambda: controller.show_frame(New_project))
        backbutton.grid(row=0, column=0, padx=10, pady=10)
    
        #############check#####################
        plot_button = ttk.Button(self.calc_options, text="Calclulate Empirical Covariance Function", command=self.plot_data)
        plot_button.grid(row=4, column=1,padx=10,pady=10)
        self.next_button=ttk.Button(self.calc_options,text="Next",command=lambda: controller.show_frame(LSC))
        self.next_button.grid(row=10,column=1,pady=30)

        self.calc_options_label=ttk.Label(self.calc_options,text="Options")
        self.calc_options_label.grid(row=0,column=1)
        self.strscale=tk.StringVar() # entry to select discance
        self.range_dist=tk.StringVar() # label variable that will show the range of the distances
        
        self.distentry=ttk.Entry(self.calc_options,textvariable=self.strscale)
        self.distentry.grid(row=2,column=1,padx=10,pady=10)
        
        

        self.dist_range=ttk.Label(self.calc_options,textvariable=self.range_dist)
        self.dist_range.grid(row=3,column=1,padx=10,pady=10)

        self.handlabel_dist=ttk.Label(self.calc_options,text="Select distance")
        self.handlabel_dist.grid(row=1,column=1,padx=10,pady=10)
        

        list_f = [ 'a*e^(-b*x) + c',
            'a*e^(-b*x) + c*e^(-d*x)',
            'a*e^(-b*x) + c*e^(-d*x) + e*e^(-x)',

            # Damped oscillation functions
            'a*e^(-b*x)*cos(c*x) + d',
            'a*e^(-b*x)*cos(c*x) + d*e^(-e*x)',
            'a*e^(-b*x)*cos(c*x) + d*e^(-e*x) + f*e^(-x)',

            # Gaussian decay functions
            'a*e^(-(x - b)^2 / (2*c^2))',
            'a*e^(-(x - b)^2 / (2*c^2)) + d',
            'a*e^(-(x - b)^2 / (2*c^2)) + d*e^(-e*x)',

            # Polynomial functions
                
        'a*x+b',
        'a*x^2+b*x+c',
        'a*x^3+b*x^2+c*x+d',
        'a*x^4+b*x^3+c*x^2+d*x+e',
        'a*x^5+b*x^4+c*x^3+d*x^2+e*x+f',
        'a*x^6+b*x^5+c*x^4+d*x^3+e*x^2+f*x+g' ,
        'a*x^7+b*x^6+c*x^5+d*x^4+e*x^3+f*x^2+g*x+h',
        'a*x^8+b*x^7+c*x^6+d*x^5+e*x^4+f*x^3+g*x^2+h*x+i',
        ]


        self.functionvar=tk.StringVar()
        self.func=ttk.Combobox(self.calc_options,textvariable=self.functionvar,width=40)
        self.func["values"]=list_f
        self.func.grid(row=7,column=1,padx=10,pady=10)

        self.functionlabel=ttk.Label(self.calc_options,text="Choose an analytical model")
        self.functionlabel.grid(row=6,column=1,padx=10,pady=10)

        self.analbutton=ttk.Button(self.calc_options,text="Calculate Analytical Model",command=self.plot_analytical_model)
        self.analbutton.grid(row=8,column=1,padx=10,pady=10)
        
        def toggle_content(notebook):
            current_tab = notebook.nametowidget(notebook.select())
            content = current_tab.winfo_children()[0]  # Assuming first child is content
            placeholder = tk.Frame(current_tab)

        self.notebook.bind("<Button-1>", lambda event: toggle_content(self.notebook))
        self.notebook2.bind("<Button-1>", lambda event: toggle_content(self.notebook2))

class LSC(tk.Frame):
    
    def lsqc(self,newlat,newlon):
       
        useless,dist,oldlat,oldlon,oldsla=self.controller.frames[New_project].minmask()
        oldsla*=100
        acc=oslaaccuracy
        f,params=self.controller.frames[ECFC].analytical_models()
      
        lat1=np.radians(newlat)
        lat2=(oldlat)
        lon1=np.radians(newlon)
        lon2=(oldlon)
        solutions=[]
        solutions_acc=[]
        covs=[]
        distances=np.empty((len(lat1),len(lat2)))
        


        

        for i in range(len(lat1)):

          
            for j in range(len(lat2)):
                dlat = lat2[j] - lat1[i]
                dlon = lon2[j] - lon1[i]
                a = np.sin(dlat / 2)**2 + np.cos(lat1[i]) * np.cos(lat2[j]) * np.sin(dlon / 2)**2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                
                R=6371
                distances[i,j]=c*R
        
    
        for i in range(len(lat1)):
    
            mask = np.where(distances[i,:] <= 3 * Cor_length_value[0])
            ddists = distances[i,mask]
            cov_obs = []
            cov_dists = dist[mask[0], :][:, mask[0]]
            cov = np.zeros_like(cov_dists)
            
            slas = oldsla[mask[0]]
            accs=acc[mask[0]]
    
          
         
       
    
            cov_obs = []
            for j in range(cov_dists.shape[0]):
             
                cov_obs.append(f(ddists[0,j], *params))  # Appending individual covariance observations
                for k in range(cov_dists.shape[0]):
               
                    cov[j, k] = f(cov_dists[j, k], *params)  # Populating the covariance matrix
                    if j==k:
                        cov[j,k]+=accs[j]


            cov_obs = np.array(cov_obs)
            
        
            if cov.shape > (0, 0):  
                cov_inv = np.linalg.inv(cov)
                
                sol=cov_obs@cov_inv@slas+meansla
                sol_acc=ypoint[0]-cov_obs@cov_inv@np.transpose(cov_obs)
                sol_accc=np.sqrt(sol_acc)
                solutions.append(sol)

                solutions_acc.append(sol_accc)
             
                covs.append(cov)
       
        return solutions,solutions_acc,covs
            
    def grid_prod(self):
        self.fpred.set("0")
        lat=[self.min_lat_entry.get(),self.max_lat_entry.get()]
        lon=[self.min_lon_entry.get(),self.max_lon_entry.get()]
        step=self.step.get()
        i=0
        delat=float(lat[1])-float(lat[0])
        delon=float(lon[1])-float(lon[0])
        latsteps=math.ceil(delat/step)
        lonsteps=math.ceil(delon/step)
       
        latpoints=np.zeros((latsteps,lonsteps))
        lonpoints=np.zeros_like(latpoints)
        for i in range(latsteps):
            for j in range(lonsteps):
                lonpoints[i,j]=j*step+float(lon[0])
                latpoints[i,j]=i*step+float(lat[0])


        nlat=latpoints.flatten(order='F')
        nlon=lonpoints.flatten(order='F')
      


        

   
        return nlat,nlon        
    
    def final_function(self):
        self.clear_plotss()
        useless,dist,oldlat,oldlon,oldsla=self.controller.frames[New_project].minmask()
        if self.fpred.get()!="0":
            nlat,nlon=self.controller.frames[LSC].read_file_predictions()
        else:
            nlat,nlon=self.controller.frames[LSC].grid_prod()

        
        oldsla*=100
        
        solution,solution_acc,cov_inv=self.controller.frames[LSC].lsqc(nlat,nlon)
    
        self.controller.frames[LSC].sla_scatter(solution,nlat,nlon)
        self.controller.frames[LSC].sla_acc_scatter(solution_acc,solution,nlat,nlon)
        self.controller.frames[LSC].cov_matrix(cov_inv[0])
        global r_lat
        global r_lon
        global r_sla
        global r_sla_acc
        global r_cov
        global r_func
        global r_par
        print(solution)
        r_func,r_par=self.controller.frames[ECFC].analytical_models()
        r_lat=nlat
        r_lon=nlon
        r_sla=solution
        r_sla_acc=solution_acc
        r_cov=cov_inv
        
    def sla_scatter(self,NSLA,newlat,newlon,**kwargs): 
        
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.set_extent([np.min(newlon) - 1, np.max(newlon) + 1,
                   np.min(newlat) - 1, np.max(newlat) + 1],
                  crs=ccrs.PlateCarree())
        
        x, y = ax.projection.transform_points(ccrs.PlateCarree(), np.array(newlon), np.array(newlat))[:, :2].T
        


        gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False  # Remove labels from the top side
        gl.right_labels = False  # Show labels on the right side
        gl.bottom_labels = True  # Show labels on the bottom side
        gl.left_labels = True # Remove labels from the left side
    
    # Plot data points on the map
        if NSLA is not None:
            scatter = ax.scatter(x, y, c=NSLA, s=10, transform=ccrs.PlateCarree(), **kwargs)
            plt.colorbar(scatter, label='Sea Level Anomaly values (cm)', cmap="warm", shrink=0.5)
        else:
            ax.scatter(x, y, transform=ccrs.PlateCarree(), **kwargs)

    # Use FigureCanvasTkAgg to embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_sla)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
    def sla_acc_scatter(self,NSLA_ACC,NSLA,newlat,newlon,**kwargs):

        
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.set_extent([np.min(newlon) - 1, np.max(newlon) + 1,
                   np.min(newlat) - 1, np.max(newlat) + 1],
                  crs=ccrs.PlateCarree())
        
        x, y = ax.projection.transform_points(ccrs.PlateCarree(), np.array(newlon), np.array(newlat))[:, :2].T


        gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False  # Remove labels from the top side
        gl.right_labels = False  # Show labels on the right side
        gl.bottom_labels = True  # Show labels on the bottom side
        gl.left_labels = True # Remove labels from the left side
    
    # Plot data points on the map
        if NSLA is not None:
            scatter = ax.scatter(x, y, c=[round(acc, 1) for acc in NSLA_ACC], s=10, transform=ccrs.PlateCarree(), **kwargs)
            plt.colorbar(scatter, label='Predicted values accuracy (cm)', cmap="warm", shrink=0.5)

        else:
            ax.scatter(x, y, transform=ccrs.PlateCarree(), **kwargs)

    # Use FigureCanvasTkAgg to embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_sla_accuracy)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def cov_matrix(self,COV_INV,**kwargs):
        
        fig = Figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(111)
        cax = ax.matshow(COV_INV, cmap='RdBu')
        ax.set_title('Covariance Matrix')

        # Add color bar
        cbar=fig.colorbar(cax)
        canvas = FigureCanvasTkAgg(fig, master=self.plot_sla_cov_matrix)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        cbar.set_label('cm\u00B2')


    def file_predictions(self):
        file_path = filedialog.askopenfilename(initialdir="~/Desktop", title="Select your prediction points")
        self.fpred.set(file_path)
      

    def read_file_predictions(self):
        if self.fpred.get()!="0":
            data = []
            with open(self.fpred.get(), "r") as file:
                for line in file:
                    data.append([float(x) for x in line.strip().split(",")])
            data_form = np.array(data)
        
        latitudes = data_form[:, 0]  # Assuming latitude is in the first column
        longitudes = data_form[:, 1]  # Assuming longitude is in the second column

        return latitudes, longitudes
        

    def predictions(self):
        self.controller.frames[LSC].file_predictions()

 

    def s_results(self, nlat, nlon, solution, solution_acc, cov_matrix):
    # Ask user to select a folder to save the files
        folder_path = filedialog.askdirectory(initialdir="~/Desktop", title="Select Folder to Save Results")
        if not folder_path:
            return

    # Define file paths for saving results
        results_file = os.path.join(folder_path, "results.txt")

        sla_res = solution
        sla_acc = solution_acc
        print(solution)

    # Convert lists to numpy arrays
        nlat = np.array(nlat)
        nlon = np.array(nlon)

    # Transpose numpy arrays
        nlat_T = np.round(nlat.T, 7)
        nlon_T = np.round(nlon.T, 7)
        sla_res = np.array(solution)
        sla_acc = np.array(solution_acc)
    # Stack arrays vertically
        print(f"nlat {np.shape(nlat)}")
        print(f"nlon {np.shape(nlon)}")
        print(f"nlat T {np.shape(nlat_T)}")
        print(f"nlon T {np.shape(nlon_T)}")




        results = np.transpose(np.vstack([nlat_T, nlon_T, np.round(sla_res.T, 2), np.round(sla_acc.T, 2)]))
    
    # Define headers for the saved file
        headers = ['Latitude (deg)', 'Longitude (deg)', 'Sea Level Anomaly (cm)', 'Sea Level Anomaly Accuracy (cm)']

    # Save results to file with default name
        np.savetxt(results_file, results, delimiter=',', header='\t'.join(headers), comments='')

    # Save covariance matrices to separate files
        for i, cov in enumerate(cov_matrix):
        # Define file path for each covariance matrix
            cov_file = os.path.join(folder_path, f"cov_matrix_{i+1}.txt")
        # Save covariance matrix to file with default name
            np.savetxt(cov_file, cov, fmt="%.25f", delimiter=",", comments="")


        


    def clear_plotss(self):
    # Clear plots from the "Midpoints vs RMS" tab
        for widget in self.plot_sla.winfo_children():
            widget.destroy()

    # Clear plots from the "Histogram of Bins" tab
        for widget in self.plot_sla_accuracy.winfo_children():
            widget.destroy()

   # Clear plots from the "Sea level anomalies plot" tab 
        for widget in self.plot_sla_cov_matrix.winfo_children():
            widget.destroy()

    

    def __init__(self, parent, controller):
        super().__init__()
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.controller = controller
        backbutton = ttk.Button(self, text="Back", command=lambda: controller.show_frame(ECFC))
        backbutton.grid(row=0,column=0)
        self.fpred=tk.StringVar()
        self.fpred.set("0")
        self.predict_options=ttk.Frame(self)
        self.predict_options.grid(row=1,column=0)

        self.plotsframe=ttk.Frame(self)
        self.plotsframe.grid(row=0,column=1,rowspan=10)
  

        self.path_option=ttk.Button(self.predict_options,text="Import Prediction Points",command=self.file_predictions)
        self.path_option.grid(row=1,column=1,pady=10)

        self.orlabel1=ttk.Label(self.predict_options,text="Or")
        self.orlabel1.grid(row=2,column=1,padx=10,pady=15)

        

        self.predict_button=ttk.Button(self.predict_options,text="Calculate",command=self.final_function)
        self.predict_button.grid(row=4,column=1)

        self.save_button=ttk.Button(self.predict_options,text="Save results",command=lambda: self.s_results(r_lat,r_lon,r_sla,r_sla_acc,r_cov))
        self.save_button.grid(row=5,column=1)


        #################################################################################33
        self.grid_frame=ttk.Frame(self.predict_options)
        self.grid_frame.grid(row=3,column=1,sticky="n")

        self.limits=ttk.Label(self.grid_frame,text="Create grid")
        self.limits.grid(row=1,column=2,columnspan=3,padx=10,pady=10)


        self.max_label=ttk.Label(self.grid_frame,text="max")
        self.max_label.grid(row=2,column=3,padx=5,pady=5)

        self.min_label=ttk.Label(self.grid_frame,text="min")
        self.min_label.grid(row=2,column=2,padx=5,pady=5)

        self.lat_label=ttk.Label(self.grid_frame,text="lat")
        self.lat_label.grid(row=3,column=0,padx=5,pady=5)

        self.lon_label=ttk.Label(self.grid_frame,text="lon")
        self.lon_label.grid(row=4,column=0,padx=5,pady=5)

        self.minlatvar=tk.DoubleVar()    
        self.min_lat_entry=ttk.Entry(self.grid_frame,textvariable=self.minlatvar)
        self.min_lat_entry.grid(row=3,column=2)

        self.maxlatvar=tk.DoubleVar()
        self.max_lat_entry=ttk.Entry(self.grid_frame,textvariable=self.maxlatvar)
        self.max_lat_entry.grid(row=3,column=3)

        self.minlonvar=tk.DoubleVar()  
        self.min_lon_entry=ttk.Entry(self.grid_frame,textvariable=self.minlonvar)
        self.min_lon_entry.grid(row=4,column=2)

        self.maxlonvar=tk.DoubleVar()  
        self.max_lon_entry=ttk.Entry(self.grid_frame,textvariable=self.maxlonvar)
        self.max_lon_entry.grid(row=4,column=3)


        self.steplabel=ttk.Label(self.grid_frame,text="Choose step")
        self.steplabel.grid(row=5,column=0,padx=5,pady=5)

        self.step=tk.DoubleVar()
        self.stepentry=ttk.Entry(self.grid_frame,textvariable=self.step)
        self.stepentry.grid(row=5,column=3)


        self.calc_button=ttk.Button(self.grid_frame,text="Calculate Grid Points",command=self.grid_prod)
        self.calc_button.grid(row=6,column=2)
        
        
    
        #################################################################################33

        self.plotbook=Notebook(self.plotsframe)
        self.plotbook.grid(row=0,column=0,rowspan=10)
        self.plotbook.grid_propagate(False)
        
        ##########plot frames#################
        self.plot_sla=ttk.Frame(self.plotbook)
        self.plot_sla_accuracy=ttk.Frame(self.plotbook)
        self.plot_sla_cov_matrix=ttk.Frame(self.plotbook)


        ############# Notebook additions ###################
        self.plotbook.add(self.plot_sla,text="Predicted SLA values")
        self.plotbook.add(self.plot_sla_accuracy,text="Prediction accuracy")
        self.plotbook.add(self.plot_sla_cov_matrix,text="Inverted matrix")
        





if __name__ == "__main__":
    app = Sea_Level_Anomaly_calculator()
    app.mainloop()

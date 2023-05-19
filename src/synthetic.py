import copy
from scipy.stats import norm
from scipy.stats import bernoulli,multinomial
import networkx as nx
import numpy as np
from dowhy import gcm
import itertools
import matplotlib.pyplot as plt
from .shapley import *
import pandas as pd

class Synthetic():
    def __init__(self) -> None:
        self.df=self.generate_data()
        self.original_df=self.df


    def generate_data(self):
        np.random.seed(1)
        x1=np.random.normal(loc=0,scale=1,size=(100))
        x2=np.random.normal(loc=0,scale=1,size=(100))
        x3= 2*x1 + 3*x2 + np.random.normal(loc=0,scale=1,size=(100))
        x4=np.random.normal(loc=0,scale=1,size=(100))

        y=x3 + 5*x4

        df=pd.DataFrame([x1,x2,x3,x4,y])
        df=df.T

        df.columns=["x1","x2","x3","x4","y"]
        return df
    
    def return_u(self):
        return 0

    def do_total_per_instance(self,x,x_s):
        
        all_fts=[-1]*len(x)
            
        if 0 in x_s:
            all_fts[0]=x[0]
        else:
            all_fts[0]=self.return_u()
            
            
        if 1 in x_s:
            all_fts[1]=x[1]
        else:
            all_fts[1]=self.return_u()
            
            
        if 2 in x_s:#populate all features 1 by 1
            all_fts[2]=x[2]
        else:
            all_fts[2]=2*all_fts[0] + 3*all_fts[1] + self.return_u() #same as directly all_FTS[2]=X[2]
            
            
        if 3 in x_s:
            all_fts[3]=x[3]
        else:
            all_fts[3]=self.return_u()
            
            
        y = all_fts[2] + 5*all_fts[3] 
        
        return y
    
    def do_total(self,xt_o,xr_o,x_s,x_sp):
        
        
        num_u=10000
        y=0
        for i in range(num_u):#iterate over u get expectation
            
            yt = self.do_total_per_instance(xt_o,x_s)
            yr = self.do_total_per_instance(xr_o,x_s)
            y = y + (yt-yr)
        
        return y/num_u
    
    

    def do_indirect(self,xt_o,xr_o,x_s,x_sp):
    #use equations known to get members out of coalition
    #then use x_s+x_p values to get y
    
        num_u=50000
        y=0
        for i in range(num_u):#iterate over u get expectation
            xt=copy.deepcopy(xt_o)
            xr=copy.deepcopy(xr_o)
            #now have people out of coalition for xt as if they were when s was as in xr
        

            if 0 in x_s:
                
                #xt[0]=return_u()
                xt[0]=xr[0]
            else:
                xt[0]=self.return_u()
                xr[0]=self.return_u()
            
            if 1 in x_s:
                #xt[1]=return_u()
                xt[1]=xr[1]
            else:
                xt[1]=self.return_u()
                xr[1]=self.return_u()
            
            if 2 in x_s:
                xt[2]=xr[2]
            else:
                if 0 in x_s:
                    t_0=xt_o[0]
                else:
                    t_0=xr[0]
                if 1 in x_s:
                    t_1=xt_o[1]
                else:
                    t_1=xr[1]
                its_u = self.return_u()
                xt[2]=2*t_0 + 3*t_1 + its_u

                if 0 in x_s:
                    t_0=xr_o[0]
                else:
                    t_0=xr[0]
                if 1 in x_s:
                    t_1=xr_o[1]
                else:
                    t_1=xr[1]

                its_u = self.return_u()
                xr[2]=2*t_0 + 3*t_1 + its_u

            if 3 in x_s:#populate all features 1 by 1
                xt[3]=xr[3]
            else:
                xt[3]=self.return_u()
                xr[3]=self.return_u()
            
            yt = xt[2] + 5*xt[3]
            yr = xr[2] + 5*xr[3]
            y = y + (yt-yr)
        return y/num_u

    def do_direct(self,xt_o,xr_o,x_s,x_sp):
    
        xt=copy.deepcopy(xt_o)
        xr=copy.deepcopy(xr_o)
        num_u=1
        y=0
        for i in range(num_u):#iterate over u get expectation
            
            if 0 in x_sp:
                
                xr[0]=self.return_u()
                xt[0]=xr[0]
            
            if 1 in x_sp:
                xr[1]=self.return_u()
                xt[1]=xr[1]
            
            if 2 in x_sp:
                t_0=xr[0]
                t_1=xr[1]

                its_u=self.return_u()
                
                xr[2]=2*t_0 + 3*t_1 + its_u
                xt[2]=xr[2]
            
            
            if 3 in x_sp:#populate all features 1 by 1
                xr[3]=self.return_u()
                xt[3]=xr[3]
            
            yt = xt[2] + 5*xt[3]
            yr = xr[2] + 5*xr[3]
            y = y + (yt-yr)
        
        return y/num_u
    
    
    
    
    def compute_shapley(self,xt,xr,kind='direct'):
        
        gc=[0,1,2,3]
        ft_map={0:"x1",1:"x2",2:"x3",3:"x4"}
        all_coalitions = list(combinations(gc))
        val={}
        
        for each_coalition in all_coalitions:
            x_s=each_coalition
            x_sp=set(gc).difference(x_s)
            
            if kind=='total':
                y=self.do_total(xt,xr,x_s,x_sp)
            elif kind=='direct':
                y=self.do_direct(xt,xr,x_s,x_sp)
            else:
                y=self.do_indirect(xt,xr,x_s,x_sp)
            
            
            val[tuple(each_coalition)]=y
        
        return val

    def get_attributions(self,xt,xr,kind='total'):
        grand_coalition=[0,1,2,3]
        ft_imp={}
        ft_map={0:"x1",1:"x2",2:"x3",3:"x4"}
        all_val_fns = self.compute_shapley(xt,xr,kind)
        #print(all_val_fns,"all_val_fns")
        for first_ft in list(grand_coalition):
            if xt[first_ft]==xr[first_ft] and kind=='direct':
                ft_imp[ft_map[first_ft]]=0
                continue
            phi_i=0
            s_wo_i=set(grand_coalition).difference(set([first_ft]))

            all_subsets=list(combinations(s_wo_i))


            for each_subset in all_subsets:
                
                v_s=all_val_fns[tuple(sorted(each_subset))]
                v_si=all_val_fns[tuple(list(sorted(set(each_subset).union(set([first_ft])))))]
                wt=coalition_wt(len(grand_coalition),len(each_subset))
                
                phi_i+=wt*(v_si-v_s)
            ft_imp[ft_map[first_ft]]=round(phi_i,2)
        
        return ft_imp
    
    def run_pair(self):
        all_shap={}
        idx_1=self.df.sample(1).index.values[0]
        idx_2=self.df.sample(1).index.values[0]

        all_shap['instance_1'] = self.df.loc[idx_1][:-1]
        all_shap['instance_2'] = self.df.loc[idx_2][:-1]
        
        for kind in ['direct','indirect','total']:
            ft_imp=self.get_attributions(self.df.values[idx_1][:-1],self.df.values[idx_2][:-1],kind)
            all_shap[kind]=ft_imp
            print(kind,ft_imp)
        print(self.original_df.loc[idx_1],self.original_df.loc[idx_2])
        return idx_1,idx_2,all_shap

    def plot(self,idx_1,idx_2,all_shap):


        # Assuming you have three lists: all_shap['direct'], all_shap['indirect'], all_shap['total']
        # Each list has 4 key-value pairs corresponding to 4 columns: S, M, R, O

        # Assuming idx_1 and idx_2 contain the row indices
        labels = ['x1', 'x2', 'x3', 'x4']
        val_labels=['x1', 'x2', 'x3', 'x4']
        row1 = self.original_df[labels].loc[idx_1].values
        row2 = self.original_df[labels].loc[idx_2].values
        labels_and_values=[val_labels[i]+"\n"+"$x_{t}$: "+ str(round(row1[i],2))+"\n"+"$x_{r}$: "+str(round(row2[i],2)) for i in range(len(labels))]
        # Create an array of indices for the bars
        indices = np.arange(len(labels))

        # Create the figure and axis objects
        fig, ax = plt.subplots()

        # Define the bar widths
        bar_width = 0.1

        # Calculate the positions of the bars
        bar_positions1 = indices - bar_width
        bar_positions2 = indices
        bar_positions3 = indices + bar_width

        # Get the values from the dictionaries for row1 and row2
        direct_values_row1 = [all_shap['direct'][col] for col in labels]
        indirect_values_row1 = [all_shap['indirect'][col] for col in labels]
        total_values_row1 = [all_shap['total'][col] for col in labels]

        direct_values_row2 = [all_shap['direct'][col] for col in labels]
        indirect_values_row2 = [all_shap['indirect'][col] for col in labels]
        total_values_row2 = [all_shap['total'][col] for col in labels]

        # Plot the stacked bar chart for row1
        ax.barh(bar_positions1, direct_values_row1, height=bar_width, align='center', label='Direct',color='brown',hatch='/')
        ax.barh(bar_positions2, indirect_values_row1, height=bar_width, align='center', label='Indirect',color='khaki')
        ax.barh(bar_positions3, total_values_row1, height=bar_width, align='center', label='Total',color='green',hatch="\\")


        font_size=5
        # Add labels to the bars in row2
        for i, val in enumerate(direct_values_row2):
            if val==0.0:
                ax.text(val, bar_positions1[i], "0", ha='left', va='center',size=font_size)
        for i, val in enumerate(indirect_values_row2):
            if val==0.0:
                ax.text(val, bar_positions2[i], "0", ha='left', va='center',size=font_size)
        for i, val in enumerate(total_values_row2):
            if val==0.0:
                ax.text(val, bar_positions3[i], "0", ha='left', va='center',size=font_size)

        # Set the y-axis ticks and labels
        ax.set_yticks(indices)
        ax.set_yticklabels(labels_and_values,size=15)

        # Set the x-axis label
        ax.set_xlabel('Shapley Values')

        # Set the legend
        ax.legend(loc='best',prop={'size': 12})
        #ax.set_xlim([0, max(max(total_values_row1), max(total_values_row2)) * 1.1])  # Adjust the multiplier as needed

        # Set the title
        #ax.set_title("Shapley attributions")
        plt.tight_layout()  # Ensures all elements are properly shown
        # Display the chart
        plt.show()


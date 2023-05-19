import copy
from scipy.stats import norm
from scipy.stats import bernoulli,multinomial
import networkx as nx
import numpy as np
from dowhy import gcm
import itertools
import matplotlib.pyplot as plt
from .shapley import *
class Compass():
    def __init__(self,clf,df,original_df) -> None:
        self.model = clf
        self.df=df
        self.original_df=original_df
        self.causal_model=None
        self.initialise_causal_model()

    def initialise_causal_model(self):
        
        causal_model = gcm.ProbabilisticCausalModel(nx.DiGraph([('C', 'A'),('C','M'),('C','Y'),('A','M'),('M','Y'),('A','Y')]))
        causal_model.set_causal_mechanism('C', gcm.ScipyDistribution(norm))
        causal_model.set_causal_mechanism('A', gcm.AdditiveNoiseModel(
            prediction_model=gcm.ml.create_linear_regressor(),
            noise_model=gcm.ScipyDistribution(norm)))
        causal_model.set_causal_mechanism('M', gcm.AdditiveNoiseModel(
            prediction_model=gcm.ml.create_linear_regressor(),
            noise_model=gcm.ScipyDistribution(norm)))
        causal_model.set_causal_mechanism('Y', gcm.AdditiveNoiseModel(
            prediction_model=gcm.ml.create_linear_regressor(),
            noise_model=gcm.ScipyDistribution(norm)))
        self.causal_model=causal_model
        gcm.fit(self.causal_model, self.df)

    def do_total_per_instance(self,x,x_s):
        
        all_fts=[-1]*len(x)
        if 1 in x_s:
            all_fts[1]=x[1]
        else:
            all_fts[1]=self.causal_model.causal_mechanism("C").draw_samples(1)[0][0]
        
        if 0 in x_s:
            all_fts[0]=x[0]
        else:
            t_0=all_fts[1]
            its_u=self.causal_model.causal_mechanism("A").noise_model.draw_samples(1)[0][0]
            all_fts[0]=self.causal_model.causal_mechanism("A").prediction_model._sklearn_mdl.predict([[t_0]])[0] + its_u
            
            
            
        if 2 in x_s:#populate all features 1 by 1
            all_fts[2]=x[2]
        else:
            t_0=all_fts[0]
            t_1=all_fts[1]
            its_u=self.causal_model.causal_mechanism("M").noise_model.draw_samples(1)[0][0]
            all_fts[2]=self.causal_model.causal_mechanism("M").prediction_model._sklearn_mdl.predict([[t_0,t_1]])[0] + its_u
        
        y=self.model.predict_proba([[all_fts[0],all_fts[1],all_fts[2]]])[0][1]
        return y
    def do_total(self,xt_o,xr_o,x_s,x_sp):
        
        
        num_u=100
        y=0
        for i in range(num_u):#iterate over u get expectation
            
            yt = self.do_total_per_instance(xt_o,x_s)
            yr = self.do_total_per_instance(xr_o,x_s)
            y = y + (yt-yr)
        
        return y/num_u
    

    def do_indirect(self,xt_o,xr_o,x_s,x_sp):
    #use equations known to get members out of coalition
    #then use x_s+x_p values to get y
        xt=copy.deepcopy(xt_o)
        xr=copy.deepcopy(xr_o)
        num_u=100
        y=0
        for i in range(num_u):#iterate over u get expectation
            
            #now have people out of coalition for xt as if they were when s was as in xr
            if 1 in x_s:
                
                #xt[0]=return_u()
                xt[1]=xr[1]
            else:
                xt[1]=self.causal_model.causal_mechanism("C").draw_samples(1)[0][0]
                xr[1]=self.causal_model.causal_mechanism("C").draw_samples(1)[0][0]
            
            if 0 in x_s:
                xt[0]=xr[0]
            else:
                if 1 in x_s:
                    t_0=xt_o[1]
                else:
                    t_0=xr[1]
                
                its_u=self.causal_model.causal_mechanism("A").noise_model.draw_samples(1)[0][0]
                xt[0]=self.causal_model.causal_mechanism("A").prediction_model._sklearn_mdl.predict([[t_0]])[0] + its_u

                if 1 in x_s:
                    t_0=xr_o[1]
                else:
                    t_0=xr[1]
                its_u = self.causal_model.causal_mechanism("A").noise_model.draw_samples(1)[0][0]
                xr[0]=self.causal_model.causal_mechanism("A").prediction_model._sklearn_mdl.predict([[t_0]])[0] + its_u
            
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
                its_u=self.causal_model.causal_mechanism("M").noise_model.draw_samples(1)[0][0]
                xt[2]=self.causal_model.causal_mechanism("M").prediction_model._sklearn_mdl.predict([[t_0,t_1]])[0] + its_u

                if 0 in x_s:
                    t_0=xr_o[0]
                else:
                    t_0=xr[0]
                if 1 in x_s:
                    t_1=xr_o[1]
                else:
                    t_1=xr[1]
                its_u = self.causal_model.causal_mechanism("M").noise_model.draw_samples(1)[0][0]
                xr[2]=self.causal_model.causal_mechanism("M").prediction_model._sklearn_mdl.predict([[t_0,t_1]])[0] + its_u

            #yt = causal_model.causal_mechanism("I").prediction_model._sklearn_mdl.predict([[xt[3],xt[2],xt[0]]])[0]
            #yr = causal_model.causal_mechanism("I").prediction_model._sklearn_mdl.predict([[xr[3],xr[2],xr[0]]])[0]
            yt=self.model.predict_proba([[xt[0],xt[1],xt[2]]])[0][1]
            yr=self.model.predict_proba([[xr[0],xr[1],xr[2]]])[0][1]
            y = y + (yt-yr)
        return y/num_u
    

    def do_direct(self,xt_o,xr_o,x_s,x_sp):
        xt=copy.deepcopy(xt_o)
        xr=copy.deepcopy(xr_o)
        num_u=100
        y=0
        for i in range(num_u):#iterate over u get expectation
            
            if 1 in x_sp:
                
                xr[1]=self.causal_model.causal_mechanism("C").draw_samples(1)[0][0]
                xt[1]=xr[1]
            
            if 0 in x_sp:
                t_0=xr[1]            
                its_u=self.causal_model.causal_mechanism("A").noise_model.draw_samples(1)[0][0]
                xr[0]=self.causal_model.causal_mechanism("A").prediction_model._sklearn_mdl.predict([[t_0]])[0] + its_u
                xt[0]=xr[0]
            
            if 2 in x_sp:
                t_0=xr[0]
                t_1=xr[1]
                its_u=self.causal_model.causal_mechanism("M").noise_model.draw_samples(1)[0][0]
                
                xr[2]=self.causal_model.causal_mechanism("M").prediction_model._sklearn_mdl.predict([[t_0,t_1]])[0] + its_u
                xt[2]=xr[2]
            
            
            #yt = causal_model.causal_mechanism("I").prediction_model._sklearn_mdl.predict([[xt[3],xt[2],xt[0]]])[0]
            #yr = causal_model.causal_mechanism("I").prediction_model._sklearn_mdl.predict([[xr[3],xr[2],xr[0]]])[0]
            yt=self.model.predict_proba([[xt[0],xt[1],xt[2]]])[0][1]
            yr=self.model.predict_proba([[xr[0],xr[1],xr[2]]])[0][1]
            y = y + (yt-yr)
        
        return y/num_u
    
    
    
    

    def compute_shapley(self,xt,xr,kind='direct'):
        
        gc=[0,1,2]
        
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
        grand_coalition=[0,1,2]
        ft_imp={}
        ft_map={0:"A",1:"C",2:"M"}
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
        print(self.model.predict_proba([self.df.loc[idx_1][:-1]])[0][1],self.model.predict_proba([self.df.loc[idx_2][:-1]])[0][1])
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
        labels = ['A', 'C', 'M']
        val_labels=['Race','Gender','Prior-Count']
        row1 = self.original_df[labels].loc[idx_1].values
        row2 = self.original_df[labels].loc[idx_2].values
        labels_and_values=[val_labels[i]+"\n"+"$x_{t}$: "+ str(row1[i])+"\n"+"$x_{r}$: "+str(row2[i]) for i in range(len(labels))]
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




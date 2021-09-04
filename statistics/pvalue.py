from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, norm, kstest
from numpy.linalg import inv
import numpy as np
import numdifftools

def get_p_value(ydata,binvals,mask=[],verbose=0,plotfile=None,yerr=None,return_teststat = False,plotsys=True,myax=None):
    ydata = np.array(ydata)
    #Assume poisson is gaussian with N+1 variance
    if not yerr:
        yerr = np.sqrt(ydata+1)
    else:
        yerr=np.array(yerr)
        
    def fit_func(x,p1,p2,p3):
        #see the ATLAS diboson resonance search: https://arxiv.org/pdf/1708.04445.pdf.
        xi = 0.
        y = x/13000.
        return p1*(1.-y)**(p2-xi*p3)*y**-p3

    xdata = np.array([0.5*(binvals[i]+binvals[i+1]) for i in range(0,len(binvals)-1)])
    xwidths = np.array([-binvals[i]+binvals[i+1] for i in range(0,len(binvals)-1)])

    #Assuming inputs are bin counts, this is needed to get densities. Important for variable-width bins
    ydata = np.array(ydata) * 100 / xwidths
    yerr = np.array(yerr)*100/ np.array(xwidths)

    #Least square fit, masking out the signal region
    popt, pcov = curve_fit(fit_func, np.delete(xdata,mask), np.delete(ydata,mask),sigma=np.delete(yerr,mask),maxfev=10000)
    if verbose:
        print('fit params: ', popt)

    ydata_fit = np.array([fit_func(x,popt[0],popt[1],popt[2]) for x in xdata])

    #Check that the function is a good fit to the sideband
    residuals = np.delete((ydata - ydata_fit)/yerr,mask)
    
    if verbose > 0:
        print("Goodness: ",kstest(residuals, norm(loc=0,scale=1).cdf))
        print(residuals)
        print(((ydata - ydata_fit)/yerr)[mask])
        print('\n')

    #The following code is used to get the bin errors by propagating the errors on the fit params
    def fit_func_array(parr):
        #see the ATLAS diboson resonance search: https://arxiv.org/pdf/1708.04445.pdf.
        p1, p2, p3 = parr
        xi = 0.
        return np.array([p1*(1.-(x/13000.))**(p2-xi*p3)*(x/13000.)**-p3 for x in xdata])
    
    jac=numdifftools.core.Jacobian(fit_func_array)
    x_cov=np.dot(np.dot(jac(popt),pcov),jac(popt).T)
    #For plot, take systematic error band as the diagonal of the covariance matrix
    y_unc=np.sqrt([row[i] for i, row in enumerate(x_cov)])

    if (plotfile != None) & (plotfile != 'ax'):
        if plotsys:
            plt.fill_between(xdata,ydata_fit+y_unc,ydata_fit-y_unc,facecolor='gray',edgecolor=None,alpha=0.4)
        yerr2 = np.array(yerr)
        yerr2[yerr>=ydata] = yerr2[yerr>=ydata]*0.8
        plt.errorbar(xdata, ydata,[yerr2,yerr],None, 'bo', label='data',markersize=4)
        plt.plot(xdata, ydata_fit, 'r--', label='data')
        plt.yscale('log', nonposy='clip')
    if plotfile == 'ax':
        if plotsys:
            myax.fill_between(xdata,ydata_fit+y_unc,ydata_fit-y_unc,facecolor='gray',edgecolor=None,alpha=0.4)
        yerr2 = np.array(yerr)
        yerr2[yerr>=ydata] = yerr2[yerr>=ydata]*0.8
        myax.errorbar(xdata, ydata,[yerr2,yerr],None, 'bo', label='data',markersize=4)
        myax.plot(xdata, ydata_fit, 'r--', label='data')
        myax.set_yscale('log', nonposy='clip')
    if plotfile == 'show':
        plt.show()
    elif plotfile:
        plt.savefig(plotfile)
        
    #Now, let's compute some statistics.
    #  Will use asymptotic formulae for p0 from Cowan et al arXiv:1007.1727
    #  and systematics procedure from https://cds.cern.ch/record/2242860/files/NOTE2017_001.pdf
    
    #First get systematics in the signal region
    
    #This function returns array of signal predictions in the signal region
    def signal_fit_func_array(parr):
        #see the ATLAS diboson resonance search: https://arxiv.org/pdf/1708.04445.pdf.
        p1, p2, p3 = parr
        xi = 0.
        return np.array([np.sum([p1*(1.-(x/13000.))**(p2-xi*p3)*(x/13000.)**-p3*xwidths[mask[i]]/100 for i, x in enumerate(xdata[mask])])])
    #Get covariance matrix of prediction uncertainties in the signal region
    jac=numdifftools.core.Jacobian(signal_fit_func_array)
    x_signal_cov=np.dot(np.dot(jac(popt),pcov),jac(popt).T)
    #Inverse signal region covariance matrix:
    inv_x_signal_cov = inv(x_signal_cov)
    
    #Get observed and predicted event counts in the signal region
    obs = np.array([np.sum(np.array(ydata)[mask]*np.array(xwidths)[mask]/100)])
    expected = np.array([np.sum([fit_func(xdata[targetbin],popt[0],popt[1],popt[2])*xwidths[targetbin]/100 for targetbin in mask])])
    
    #Negative numerator of log likelihood ratio, for signal rate mu = 0
    def min_log_numerator(expected_nuis_arr):
        #expected_nuis_arr is the array of systematic background uncertainty nuisance parameters
        #These are event rate densities
        expected_nuis_arr = np.array(expected_nuis_arr)
        to_return = 0
        #Poisson terms
        for i, expected_nuis in enumerate(expected_nuis_arr):
            #Poisson lambda. Have to rescale nuisance constribution by bin width
            my_lambda = expected[i]+expected_nuis_arr[i]
            #Prevent negative predicted rates
            if my_lambda < 10**-10:
                my_lambda = 10**-10
            #Poisson term. Ignore the factorial piece which will cancel in likelihood ratio
            to_return = to_return + (obs[i]*np.log(my_lambda) - my_lambda)
            
        #Gaussian nuisance term
        nuisance_term = -0.5*np.dot(np.dot(expected_nuis_arr,inv_x_signal_cov),expected_nuis_arr)
        to_return = to_return + nuisance_term
        return -to_return

    def jac_min_log_numerator(expected_nuis_arr):
        #expected_nuis_arr is the array of systematic background uncertainty nuisance parameters
        #These are event rate densities
        expected_nuis_arr = np.array(expected_nuis_arr)
        to_return = np.array([0.])
        #Poisson terms
        #Poisson lambda. Have to rescale nuisance constribution by bin width
        my_lambda = expected+expected_nuis_arr
        dmy_lambda = np.array([1.])
        #Prevent negative predicted rates
        my_lambda[my_lambda < 10**-10] = np.ones(len(my_lambda[my_lambda < 10**-10])) * 10**-10
        dmy_lambda[my_lambda < 10**-10] = 0
        #Poisson term. Ignore the factorial piece which will cancel in likelihood ratio
        to_return = to_return + (obs*dmy_lambda/my_lambda - dmy_lambda)
        #Gaussian nuisance term
        nuisance_term = -np.dot(inv_x_signal_cov,expected_nuis_arr)
        to_return = to_return + nuisance_term
        return -to_return
    
    #Initialization of nuisance params
    expected_nuis_array_init = [0.02]
    
    #shift log likelihood to heklp minimization algo
    def rescaled_min_log_numerator(expected_nuis_arr):
        return min_log_numerator(expected_nuis_arr) - min_log_numerator(expected_nuis_array_init)
    
    #Perform minimization over nuisance parameters. Set bounds for bg nuisance at around 8 sigma.
    bnds=[[-8*y_unc[mask[0]],8*y_unc[mask[0]]]]
    minimize_log_numerator = minimize(rescaled_min_log_numerator,
                                      expected_nuis_array_init,
                                      jac=jac_min_log_numerator,
                                      bounds=bnds)
    
    if verbose:
        print("numerator: ",  minimize_log_numerator.items(),'\n')
        
    #Now get likelihood ratio denominator
    def min_log_denom(nuis_arr):
        #nuis_arr contains the bg systematics and also the signal rate
        expected_nuis_arr = np.array(nuis_arr)[:1]
        #print(expected_nuis_arr)
        mu = nuis_arr[1]
        #Signal prediction
        pred = [mu]
        to_return = 0
        #Poisson terms
        for i, expected_nuis in enumerate(expected_nuis_arr):
            #Poisson lambda
            my_lambda = expected[i]+expected_nuis_arr[i] + pred[i]
            #Prevent prediction from going negative
            if my_lambda < 10**-10:
                my_lambda = 10**-10
            #Poisson term. Ignore the factorial piece which will cancel in likelihood ratio
            to_return = to_return + (obs[i]*np.log(my_lambda) - my_lambda)

        #Gaussian nuisance term
        nuisance_term = -0.5*np.dot(np.dot(expected_nuis_arr,inv_x_signal_cov),expected_nuis_arr)
        to_return = to_return + nuisance_term
        return -to_return

    def jac_min_log_denom(nuis_arr):
        #expected_nuis_arr is the array of systematic background uncertainty nuisance parameters
        #These are event rate densities
        expected_nuis_arr = np.array(nuis_arr)[:1]
        mu = nuis_arr[1]
        pred = [mu]
        to_return_first = np.array([0.])
        #Poisson terms
        #Poisson lambda. Have to rescale nuisance constribution by bin width
        my_lambda = expected+expected_nuis_arr+pred
        dmy_lambda = np.array([1.])
        #Prevent prediction from going negative
        my_lambda[my_lambda < 10**-10] = np.ones(len(my_lambda[my_lambda < 10**-10])) * 10**-10
        dmy_lambda[my_lambda < 10**-10] = 0
        #Poisson term. Ignore the factorial piece which will cancel in likelihood ratio
        to_return_first = to_return_first + (obs*dmy_lambda/my_lambda - dmy_lambda)
        #Gaussian nuisance term
        nuisance_term = -np.dot(inv_x_signal_cov,expected_nuis_arr)
        to_return_first = to_return_first + nuisance_term
        
        to_return_last = np.array([0.])
        
        dpred = np.array([[1.]])
        
        my_lambda = expected+expected_nuis_arr+pred
        dmy_lambda = dpred
        to_return_last = np.dot((obs/my_lambda),dmy_lambda.T) - np.sum(dmy_lambda,axis=1)
        
        return -np.append(to_return_first, to_return_last)
    
    #initizalization for minimization
    nuis_array_init = [0.01,1.]
    
    #Shift log likelihood for helping minimization algo.
    def rescaled_min_log_denom(nuis_arr):
        return min_log_denom(nuis_arr) - min_log_denom(nuis_array_init)
    
    bnds = ((None,None),(None,None))
    minimize_log_denominator = minimize(rescaled_min_log_denom,nuis_array_init,
                                        jac=jac_min_log_denom,
                                        bounds=bnds)
    
    if verbose:
        print("Denominator: ",  minimize_log_denominator.items(),'\n')
        
    if minimize_log_denominator.x[-1] < 0:
        Zval = 0
        neglognum = 0
        neglogden = 0
    else:
        neglognum = min_log_numerator(minimize_log_numerator.x)
        neglogden = min_log_denom(minimize_log_denominator.x)
        Zval = np.sqrt(2*(neglognum - neglogden))
      
    
    p0 = 1-norm.cdf(Zval)
    
    if verbose:
        print("z = ", Zval)
        print("p0 = ", p0)

    #plt.title(str(p0))
#     if plotfile == 'show':
#         plt.show()
#     elif plotfile:
#         plt.savefig(plotfile)

    if return_teststat:
        return p0, 2*(neglognum - neglogden)
    else:
        return p0

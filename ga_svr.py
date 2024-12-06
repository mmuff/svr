# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

class GA_CV(BaseEstimator, TransformerMixin):
    import numpy as np
    
    
    def __init__(self, iters = 100, mut_rat = .05, pop_size = 100, n_var = 4, 
                 eps=0.2, kernel='linear', C=1, degree=3,
                 ran_rat = .2, random_state = None, verbose = 0, 
                 scale_data = True, scorer = 'r2', model = 'svr',
                 corr_tresh = .85, cv = 3, max_loops = 100): # , max_loops = 50
        self.iters = iters
        self.mut_rat = mut_rat
        if pop_size % 2 == 1:
            pop_size -= 1
        self.pop_size = pop_size
        self.n_var = n_var
        self.ran_rat = ran_rat
        self.random_state = random_state
        self.verbose = verbose
        self.scale_data = scale_data
        self.corr_tresh = corr_tresh
        self.best_score = []
        self.max_loops = max_loops
        self.variables = []
        self.cv = cv
        self.max_loops = max_loops
        self.eps = eps
        self.kernel = kernel
        self.C = C
        self.degree = degree
        if scorer == 'r2':
            from sklearn.metrics import r2_score
            self.scorer = r2_score
        else:
            self.scorer = scorer
        if model == 'svr':
            
            self.model = SVR(epsilon=self.eps, kernel=self.kernel, max_iter=-1,
                             C=self.C, degree=self.degree)
        else:
            self.model = model
        
    
    def fit(self, X, y , X_names):
        import warnings
        warnings.filterwarnings("ignore")
        import random
        from sklearn.model_selection import cross_val_predict
        
        def cross_over(a,b,mut,X):
            a = np.array(a)
            b = np.array(b)
            a1 = a.copy()
            b1 = b.copy()
            mut_c = int(np.ceil(a.shape[0] * mut))
            tem = np.zeros(a.shape[0])
            ads = np.random.randint(1,a.shape[0],mut_c)
            lis_tmp =  []
            ad_ran = range(len(a))
            for it in range(len(ads)):
                bb = np.array([xi for i,xi in enumerate(ad_ran) if i!=ads[it]])
                lis_tmp.append(b[ads[it]] in a[bb])
                lis_tmp.append(a[ads[it]] in b[bb])
            if any(lis_tmp):
                pass
            else:
                tem[ads] = a[ads]
                a[ads] = b[ads]
                b[ads] = tem[ads]
                a = a.astype(int).tolist()
                b = b.astype(int).tolist()
                a = np.array(a)
                b = np.array(b)
            if (np.abs(np.tril(np.corrcoef(X[0:,a[:]],rowvar=False),k=-1)[0:,0:]) >= self.corr_tresh).any() :
                a = a1.copy()
                b = b1.copy()
            
            elif (np.abs(np.tril(np.corrcoef(X[0:,b[:]],rowvar=False),k=-1)[0:,0:]) >= self.corr_tresh).any():
                a = a1.copy()
                b = b1.copy()
            else:
                pass
            a = a.astype(int).tolist()
            b = b.astype(int).tolist()
            return a, b
        
        if self.cv == False:
            def score_it_cv(X,y,mask):
                scores = []
                for i in range(mask.shape[0]):
                    self.model.fit(X[:,mask[i]],y)
                    y_p = self.model.predict(X[:,mask[i]])
                    if any(np.abs(np.corrcoef(X[:,mask[i]],rowvar=False)[0,1:]) >= self.corr_tresh):
                        scores.append(0)
                    else:
                        scores.append(self.scorer(y,y_p))
                scores = np.array(scores).reshape((-1,1))
                return scores
        else:
            def score_it_cv(X,y,mask):
                scores = []
                for i in range(mask.shape[0]):
                    if any(np.abs(np.corrcoef(X[:,mask[i]],rowvar=False)[0,1:]) >= self.corr_tresh):
                        scores.append(0)
                    else:
                        y_p = cross_val_predict(estimator = self.model,
                                        X=X[:,mask[i]], y=y, cv=self.cv)
                        scores.append(self.scorer(y,y_p))
                scores = np.array(scores).reshape((-1,1))
                return scores

        sizeX = X.shape[1]
        if self.random_state != None:
            np.random.seed(self.random_state)
        if self.scale_data == True:
            from sklearn.preprocessing import StandardScaler
            st_sc = StandardScaler()
            X = st_sc.fit_transform(X)
        else:
            X = np.array(X)
        
        # initial state - random mask and its scores
        mask = []
        while len(mask) < self.pop_size:
            if len(mask) == self.pop_size:
                break
            ran = random.sample(list(np.arange(sizeX)),int(self.n_var))
            if ran not in mask and not (np.abs(np.tril(np.corrcoef(X[0:,ran[:]],rowvar=False),k=-1)[0:,0:]) >= self.corr_tresh).any():
                mask.append(ran)
            else:
                ite = 0
                while ite < self.max_loops:
                    ite = ite + 1
                    ran = random.sample(list(np.arange(sizeX)),int(self.n_var))
                    if ran not in mask and not (np.abs(np.tril(np.corrcoef(X[0:,ran[:]],rowvar=False),k=-1)[0:,0:]) >= self.corr_tresh).any():
                        mask.append(ran)
                        break
                    
                    if ite == self.max_loops:
                        if self.pop_size > 10:
                            self.pop_size = self.pop_size - 2
                        else:
                            self.corr_tresh = self.corr_tresh + 0.01
        print('Population size is now: ', self.pop_size)
        print('Correlation tresh is now: ', self.corr_tresh)
        
        mask = np.array(mask)
        reshap = False
        while not reshap:
            try:
                mask.reshape((self.pop_size,self.n_var))
                reshap = True
            except:
                mask = mask[:-1]
        scores = score_it_cv(X,y,mask)
        master = np.concatenate((mask,scores),axis=1)
        master = master[master[:,-1].argsort()[::-1]]
        
        cur_best = master[0,-1:]
        if self.verbose > 2:
            print('Initial (random) best fit is: {0:.6f}'.format(float(cur_best)))
        
        cut_off = int(np.ceil((1 - self.ran_rat) * self.pop_size))
        #from IPython.display import clear_output
        for it in range(self.iters):
#             if it % 100 == 0:
#                 print('Przeliczono {} iteracji'.format(it))
            temp = master[:cut_off,:-1].astype(int).tolist()
            while len(temp) < self.pop_size:
                ran = random.sample(list(np.arange(sizeX)),int(self.n_var))
                if not (np.abs(np.tril(np.corrcoef(X[0:,ran[:]],rowvar=False),k=-1)[0:,0:]) >= self.corr_tresh).any():
                    temp.append(ran)
                    
            pairs = np.random.permutation(np.arange(len(temp))).reshape((-1,2))
            for l,p in enumerate(pairs):
                temp[p[0]], temp[p[1]] = cross_over(temp[p[0]],temp[p[1]],self.mut_rat,X)

            temp = np.array(temp)
            temp.reshape((self.pop_size,self.n_var))
            scores_t = score_it_cv(X,y,temp)
            master_t = np.concatenate((temp,scores_t),axis=1)
            master = np.concatenate((master,master_t),axis=0)
                        
            master = np.unique(master,axis=0)
            master = master[master[:,-1].argsort()[::-1]]
            master = master[:self.pop_size,:]
            self.variables = master[0,:-1].astype(int).tolist()
            if self.verbose == 5:   
                print('')
                print('Iteration: ', it+1)
                print('Current best score: {0:.3f}'.format(float(master[0,-1:])))
                print('Set of variables:', X_names[self.variables])
                #clear_output(wait=True)
                #print('Indices of your variables: ', self.variables)
        
        self.best_score = float(master[0,-1:])
        self.variables = master[0,:-1].astype(int).tolist()
        
        if self.verbose > 2:
            print('')
            print('-------')
            print('The best score: {0:.6f}'.format(float(master[0,-1:])))
            print('Selected variables:', X_names[self.variables])
            print('-------')
        warnings.filterwarnings("default")
        self.mask = mask

    def transform(self, X):
        if not not self.variables:
            X = X.iloc[:,self.variables]
            if self.scale_data == True:
                from sklearn.preprocessing import StandardScaler
                st_sc = StandardScaler()
                X = st_sc.fit_transform(X)
            else:
                X = np.array(X)
        else:
            print('X not affected\nUse the fit method first')
        return X

    def fit_transform(self,X,y):
        self.fit(X,y)
        X = self.transform(X)
        return X
        
    def predict(self,X,y):
        self.fit(X,y)
        return self.best_score

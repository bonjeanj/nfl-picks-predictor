def expected(a,b): return 1/(1+10**((b-a)/400))
def update_elo(a,b,r,k=20):
 e=expected(a,b);return a+k*(r-e), b+k*((1-r)-(1-e))

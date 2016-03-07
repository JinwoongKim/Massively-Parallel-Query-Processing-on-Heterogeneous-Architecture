#define range_3(i,a,b)     i = (a); i < (b); ++i
#define range_4(i,a,b,j)   i = (a); i < (b); i+=j

#define range_X(x,i,a,b,j,FUNC, ...)  FUNC  

#define range(...)          range_X(,##__VA_ARGS__,\
                            range_4(__VA_ARGS__),\
                            range_3(__VA_ARGS__)\
                            ) 

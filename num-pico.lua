-- num-pico lib
-- returns a n-by-m array with random values (-5,5)
function np_mat_rand(_n,_m)
    local a={}
    for i=1,_n do -- each row
        a[i]={}
        for j=1,_m do -- each col
            a[i][j]=rnd(10)-5
        end
    end
    return a
end
-- returns a n-dim array with random values (-5,5)
function np_vec_rand(_n)
    local a={}
    for i=1,_n do
        a[i]=rnd(10)-5
    end
    return a
end
-- dot product of 2 matrices, a & b
-- a is m-by-n, b is n-by-p
-- returns m-by-p matrix
function np_dot(_a,_b)
    local m=#_a
    local n=#_b
    local p=#_b[1]
    local c={}
    if (#_a[1] != n) then -- proper dims check
        return -- cannot compute
    else
        -- compute result matrix
        for i=1,m do -- each row
            c[i]={}
            for j=1,p do -- each col
                c[i][j]=0
                for k=1,n do -- each n
                    c[i][j]+=_a[i][k] * _b[k][j]
                end
            end
        end
    end
    return c
end
-- dot product of 1-d vector and n-by-p matrix
-- a is array of legnth n (1-by-n transposed)
-- b is n-by-p matrix
-- returns array of length p (1-by-p transposed)
function np_vm_dot(_a,_b)
    local n=#_a
    local p=#_b[1]
    local c={}
    -- compute result matrix
    for i=1,p do -- each col (b)
        c[i]=0
        for j=1,n do -- each row (b)
            c[i]+=_a[j]*_b[j][i]
        end
    end
    return c
end
-- dot product of n-by-p matrix and 1-d vector
-- a is n-by-p matrix
-- b is array of legnth p (1-by-p transposed)
-- returns array of length n (1-by-n transposed)
function np_mv_dot(_a,_b)
    local n=#_a
    local p=#_a[1]
    local c={}
    -- compute result matrix
    for i=1,n do -- each row (a)
        c[i]=0
        for j=1,p do -- each col (a)
            c[i]+=_b[j]*_a[i][j]
        end
    end
    return c
end
-- component-wise vector addition
-- a is 1-d array of length n
-- b is 1-d array of length n
-- returns n-d array
function np_vec_add(_a,_b)
    local c={}
    for i=1,#_a do
        c[i]=_a[i]+_b[i]
    end
    return c
end
-- component-wise vector subtraction
-- a is 1-d array of length n
-- b is 1-d array of length n
-- returns n-d array
function np_vec_sub(_a,_b)
    local c={}
    for i=1,#_a do
        c[i]=_a[i]-_b[i]
    end
    return c
end
-- component-wise vector multiplication
-- a is 1-d array of length n
-- b is 1-d array of length n
-- returns n-d array
function np_vec_mult(_a,_b)
    local c={}
    for i=1,#_a do
        c[i]=_a[i]*_b[i]
    end
    return c
end
-- matrix component-wise addtion
-- a is m-by-n, b is n-by-p
-- returns array of length m
function np_add(_a,_b)
    local m=#_a
    local n=#_b
    local p=#_a[1]
    local c={}
    -- compute result matrix
    for i=1,m do -- each row
        c[i]={}
        for j=1,p do -- each col
            c[i][j]=_a[i][j] + _b[i][j]
        end
    end
    return c
end
-- component-wise x^2
function np_vec_sq(_v)
    for comp in all(_v) do
        comp = comp*comp
    end
    return _v
end
-- returns: arithmetic mean of vector
function np_vec_mean(_v)
    local sum=0
    for i=1,#_v do
        sum+=_v[i]
    end
    return sum/#_v
end
-- applies function to each vector component
function np_vec_func(_v,_func)
    for i=1,#_v do
        _v[i]=_func(_v[i])
    end
    return _v
end
-- returns index of max value in array
function np_argmax(vec)
    local max=-1
    local indx=0
    for i=1,#vec do
        if vec[i]>max then
            max=vec[i]
            indx=i
        end
    end
    return indx
end
-- sigmoid
function np_sigmoid(x)
    return 1 / (1+np_exp(-x))
end
-- derived sigmoid
function np_deriv_sigmoid(x)
    return x*(1-x)
end
-- hiberbolic tangent
function np_tanh(x)
	return np_sinh(x)/np_cosh(x)
end
-- derived sigmoid
function np_deriv_tanh(x)
    return 1-(x^2)
end
-- hiberbolic cosine
function np_cosh(x)
	return 0.5*(np_exp(x)-np_exp(-x))
end
-- hiberbolic sine
function np_sinh(x)
	return 0.5*(np_exp(x)+np_exp(-x))	
end
-- exponent
function np_exp(x)
	local e=2.71828183
	return e^x
end
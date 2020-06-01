pico-8 cartridge // http://www.pico-8.com
version 27
__lua__
-- pico learn
-- by blearn
function _init()
    draw=true
    rectfill(0,0,127,127,0)
    --
    --net = init_graph()
    
    --net = fcnn:new({3,5,5,3})
    --draw_nn(net)
   

    num_inputs=4
    num_outputs=6
    net=nn:new()
    net:add_layer(layer:new(num_inputs,3,"tanh"))
    net:add_layer(layer:new(3,2,"sigmoid"))
    net:add_layer(layer:new(2,3,"sigmoid"))
    net:add_layer(layer:new(3,num_outputs,"sigmoid"))

    --num_inputs=2
    --num_outputs=2
    --net:add_layer(layer:new(num_inputs,3,"tanh"))
    --net:add_layer(layer:new(3,3,"sigmoid"))
    --net:add_layer(layer:new(3,num_outputs,"sigmoid"))
    
    --num_inputs=4
    --num_outputs=6
    --net:add_layer(layer:new(num_inputs,3,"tanh"))
    --net:add_layer(layer:new(3,3,"sigmoid"))
    --net:add_layer(layer:new(3,num_outputs,"sigmoid"))

    output={}
    output=net:feedforward(np_rand_vec(num_inputs))

    -- configure training
    learing_rate=0.3
    epochs = 10000
    num_samples = 100

    -- generate random test data
    x = {}
    y = {}
    for s=1,num_samples do -- each input sample
        x[s]={}
        for i=1,num_inputs do -- each input
            x[s][i]=rnd()
        end
        y[s]={}
        for l=1,num_outputs do -- each label
            y[s][l]=rnd()
        end
    end

    -- define dataset for logical 'and'
    --x={{0,0},{0,1},{1,0},{1,1}}
    --y={{1,0},{1,0},{1,0},{0,1}}


    -- define simple dataset (in:2, out:2)
    --x={{0,0},{0,1},{1,0},{1,1}}
    --y={{0,0},{0,1},{1,0},{1,1}}


    -- define simple dataset (in:4 out: 6)
    -- x={{0,0,0,0},{0,0,0,1},{0,0,1,0},{0,0,1,1}}
    -- y={{0,0,0,0,0,0},{0,0,0,0,0,1},{0,0,0,0,1,0},{0,0,0,0,1,1}}


    num_samples=4

    prev_sample=1
    prev_epoch=-1

    errors={}
    done_training=false
    -- train
    --net:train(x,y,learing_rate,epochs)
end
function _update()
    --output=net:feedforward(np_rand_vec(num_inputs))

    -- train
    if ((net.epoch>prev_epoch) and (net.epoch<epochs)) then -- ready for next epoch
        prev_epoch=net.epoch
        net:train_step(x[prev_sample],y[prev_sample],learing_rate)
        add(errors, net:mse(y[1],x[1]))
        if (prev_sample<num_samples) then -- continue batch
            prev_sample+=1
        else -- batch complete
            prev_sample=1
            net:accuracy(x, y)
        end
    elseif (net.epoch>=epochs and not(done_training)) then -- training over
        done_training=true
        net:accuracy(x, y)
    end
end
function _draw()
    if (draw) then
        rectfill(0,0,127,127,0)
        rect(0,0,127,127,6)
        net:draw()
        --print("output: ", 16,64, 6)
        --for i=1,#output do
        --    print(tostr(output[i]), 32,72+(i*8), 6)
        --end
    end
    --
    --draw_nn(net)
    --draw_graph(net)
end
-->8
-- setup graph
function init_graph()
    --
    local n=node:new()
    n:setpos(10,10)
    n:connect(n:new(20,10))
    n:connect(n:new(20,30))
    n:connect(n:new(20,40))
    return n
end
-->8
-- demo
-->8
-- graph lib
graph_lib = {}
--graph_lib.heat_pal={1,2,8,14,7} -- heat index palette
--graph_lib.heat_pal={2,4,9,11,12} -- heat index palette
--graph_lib.heat_pal={12,11,10,9,8} -- heat index palette
graph_lib.heat_pal={4,9,7,12,11} -- "brbg" heat index palette
node={}
-- node constructor
function node:new(_x, _y)
    local this={}
    if _x==nil then this.x=0 else this.x=_x end
    if _y==nil then this.y=0 else this.y=_y end
    this.edges={}
    this.value=rnd(4)+1

    self.__index=self
    setmetatable(this,self)
    return this
end
edge={}
-- edge constructor
function edge:new(_srcnode, _dstnode)
    local this={}
    this.src=_srcnode
    this.dst=_dstnode
    this.weight=rnd(4)+1

    self.__index=self
    setmetatable(this,self)
    return this
end
function node:connect(_node) -- uni-directional connection to node
    local e = edge:new(self,_node) -- self -> _node
    add(self.edges,e)
end
function node:disconnect(_edge) -- sever uni-directional connection
    del(self.edges,_edge)
end
function node:setpos(_x, _y)
    self.x=_x
    self.y=_y
end
function node:setval(_val)
    self.value=_val
end

-- only safe with DAGs (no loops)
function draw_graph(root)
    if (root != nil) then
        -- bfs
        for e in all(root.edges) do
            local c=graph_lib.heat_pal[mid(1,flr(e.weight),5)]
            line(e.src.x,e.src.y, e.dst.x,e.dst.y, c)
            draw_graph(e.dst)
        end
        local c=graph_lib.heat_pal[mid(1,flr(root.value),5)]
        circfill(root.x,root.y,3, c)
        circ(root.x,root.y,3, 7)
    end
end

-->8
-- reinforcement learning lib
nn={}
-- constructor for fully-connected neural network
function nn:new()
    local this={}
    this.layers={}
    this.epoch=0
    this.error=0.0
    this.acc=0.0

    self.__index=self
    setmetatable(this,self)
    return this
end
function nn:add_layer(layer)
    add(self.layers, layer)
end
-- feed forward input through layers
-- x: n-d array
-- return: n-d array
function nn:feedforward(x)
    for layer in all(self.layers) do
        x=layer:activate(x)
    end
    return x
end
-- predicts a class
-- useful with sigmoid activation (interpret outputs as probabilities)
-- returns index of predicted classed
function nn:predict(x)
    local ff=self:feedforward(x)
    return np_argmax(ff)
end
-- backpropagation
-- uses mean squared sum loss function
-- x: input values (array)
-- y: target values (array)
-- lr: learning rate (0, 1)
function nn:backprop(x,y,lr)
    -- feedforward
    local output=self:feedforward(x)

    -- calculate errors and deltas
    for l=#self.layers,1,-1 do -- loop over layers backwards
        local layer=self.layers[l]
        if l==#self.layers then -- if this is output layer
            layer.error=np_sub_vec(y,output)
            layer.delta=np_mult_vec(layer.error,layer:apply_activation_derivative(output))
        else
            local next_layer=self.layers[l+1]
            layer.error=np_dot_mv(next_layer.weights, next_layer.delta)
            layer.delta=np_mult_vec(layer.error,layer:apply_activation_derivative(layer.last_activation))
        end
    end

    -- update weights
    for l=1,#self.layers do -- loop over layers
        local _in={}
        if (l!=1) then -- input is previous layer
            _in=self.layers[l-1].last_activation
        else -- input is x
            _in=x
        end
        -- apply update: w += delta*x*lr
        self.layers[l]:update(_in,self.layers[l].delta,lr)
    end
end
-- accuracy between predicted and true labels
function nn:accuracy(y_pred,y_true)
    local mean=0
    for i=1,#y_pred do -- each sample
        if (nn:predict(y_pred[i])==np_argmax(y_true[i])) then mean+=1 end
    end
    self.acc=mean/#y_pred
    return self.acc
end
-- mean square error
-- y: input vector, 1-d array
-- returns: mse, scalar
function nn:mse(y,x)
    local e=np_mean_vec(np_square_vec(np_sub_vec(y, self:feedforward(x))))
    self.error=e
    return e
end
-- train using sgd
-- x: input values, 2-d array. x[1] is the first sample vector
-- y: target values, 2-d array. y[1] is the first target vector
-- lr: learning rate (0,1)
-- max_epochs: maximum number of training iterations
-- returns: array of mse errors
function nn:train(x,y,lr,max_epochs)
    local results={}
    for i=1,max_epochs do -- each epoch
        self.epoch=i
        for s=1,#x do -- each sample
            self:backprop(x[s], y[s], lr)
        end
        -- check progress
        if i%10==0 then
            add(results, nn:mse(y[1],x[1]))
        end
    end
    return results
end
-- train on one sample using sgd
-- x: input values, 1-d array. x is a sample vector
-- y: target values, 1-d array. y is a target vector
-- lr: learning rate (0,1)
-- max_epochs: maximum number of training iterations
function nn:train_step(x,y,lr)
    self.epoch+=1
    self:backprop(x, y, lr)
end
-- draw nn structure
function nn:draw()
    local radius=3
    local pad=16
    for lindx,layer in ipairs(self.layers) do -- each layer
        local w=layer.weights
        local ins=#w
        local outs=#w[1]
        -- draw weights
        for r=1,ins do
            local x=pad
            local y=r*pad
            for c=1,outs do
                x=lindx*pad
                local v=(w[r][c]+5)/2 -- map value (-5,5) to (1,5)
                local h=graph_lib.heat_pal[mid(1,flr(v),5)]
                local x1=x+pad
                local y1=c*pad
                line(x,y, x1,y1, h)
            end
        end
        -- draw neurons
        -- if lindx==1 then -- first layer
        for r=1,ins do -- draw input neurons
            circfill(lindx*pad,r*pad, radius, graph_lib.heat_pal[1])
            circ(lindx*pad,r*pad, radius, 7)
        end
        for c=1,outs do -- draw output neurons
            local v=self.layers[lindx].last_activation[c]
            if (self.layers[lindx].activation=="tanh") then
                -- map value (-1,1) to (1,5)
                v=(v+1)*(5/2)
            else -- sigmoid
                -- map value (0,1) to (1,5)
                v=v*5
            end
            local h=graph_lib.heat_pal[mid(1,flr(v),5)]
            circfill((lindx+1)*pad,c*pad, radius, h)
            circ((lindx+1)*pad,c*pad, radius, 7)
        end
    end
    -- draw heat index
    local hi_x=116
    local hi_y=84
    local hi_w=8
    local hi=5
    for i=1,#graph_lib.heat_pal do
        rectfill(hi_x,hi_y+((i-1)*hi_w), hi_x+hi_w,hi_y+((i-1)*hi_w)+hi_w, graph_lib.heat_pal[hi])
        hi-=1
    end
    print("max", 100,85, graph_lib.heat_pal[5])
    print("min", 100,117, graph_lib.heat_pal[1])
    -- draw train stats
    print("epoch: "..self.epoch, 15,108, 6)
    print("error: "..self.error, 15,114, 13)
    print("accuracy: "..(self.acc*100).."%", 3,120, 15)
end
layer={}
-- constructor for nn layer
function layer:new(n_input, n_neurons, activation, weights, bias)
    local this={}
    if activation != nil then this.activation=activation else
        this.activation="tanh"
    end
    if weights != nil then this.weights=weights else
        this.weights=np_rand_mat(n_input,n_neurons)
    end
    if bias != nil then this.bias=bias else 
        this.bias=np_rand_vec(n_neurons)
    end


    self.__index=self
    setmetatable(this,self)
    return this
end
-- calculates dot product of this layer
-- x is 1-d array (inputs)
-- returns vector: XW+B
function layer:activate(x)
    --print("x: "..#x)
    local res=np_dot_vm(x, self.weights) -- X dot W
    --print("xw: "..#res)
    res=np_add_vec(res, self.bias) -- add bias
    --print("xw+b: "..#res)
    self.last_activation=self:apply_activation(res) -- apply activiation fn
    --print("fn(xw+b): "..#self.last_activation)
    return self.last_activation
end
function layer:apply_activation(vec)
    if self.activation=="tanh" then
        vec=np_vec_func(vec, np_tanh)
    else -- assume "sigmoid"
        vec=np_vec_func(vec, np_sigmoid)
    end
    return vec
end
function layer:apply_activation_derivative(vec)
    if self.activation=="tanh" then
        vec=np_vec_func(vec, np_deriv_tanh)
    else -- assume "sigmoid"
        vec=np_vec_func(vec, np_deriv_sigmoid)
    end
    return vec
end
function layer:update(_input,_output,_lr)
    for i=1,#_input do
        for o=1,#_output do
            local d = _output[o]*_input[i]*_lr
            self.weights[i][o]=mid(-5, self.weights[i][o]+d, 5) -- clamp (-5, 5)
            --self.weights[i][o]+=d
        end
    end
end
-->8
-- num-pico lib
-- returns a n-by-m array with random values (-5,5)
function np_rand_mat(_n,_m)
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
function np_rand_vec(_n)
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
function np_dot_vm(_a,_b)
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
function np_dot_mv(_a,_b)
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
function np_add_vec(_a,_b)
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
function np_sub_vec(_a,_b)
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
function np_mult_vec(_a,_b)
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
function np_square_vec(_v)
    for comp in all(_v) do
        comp = comp*comp
    end
    return _v
end
-- returns: arithmetic mean of vector
function np_mean_vec(_v)
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

__gfx__
00000000700303080000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000e0010c090000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00700700800c070a0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00077000200509030000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00077000100404010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00700700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

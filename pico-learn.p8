pico-8 cartridge // http://www.pico-8.com
version 27
__lua__
-- pico learn
-- by blearn
function _init()

    rectfill(0,0,127,127,0)
    --
    --net = init_graph()
    
    --net = fcnn:new({3,5,5,3})
    --draw_nn(net)
    num_inputs=2
    num_outputs=6

    net=nn:new()
    net:add_layer(layer:new(num_inputs,3,"tanh"))
    net:add_layer(layer:new(3,3,"sigmoid"))
    net:add_layer(layer:new(3,num_outputs,"sigmoid"))

    output={}
end
function _update()
    output=net:feedforward(np_rand_vec(num_inputs))
end
function _draw()
    rectfill(0,0,127,127,0)
    net:draw()
    print("output: ", 16,64, 6)
    for i=1,#output do
        print(tostr(output[i]), 32,72+(i*8), 6)
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
graph_lib.heat_pal={1,2,8,14,7} -- heat index palette
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
    local ff=self.feedforward(x)
    return np_argmax(ff)
end
-- draw nn structure
function nn:draw()
    local radius=2
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
                local h=graph_lib.heat_pal[mid(1,flr(5*w[r][c]),5)]
                local x1=x+pad
                local y1=c*pad
                line(x,y, x1,y1, h)
            end
        end
        -- draw neurons
        for r=1,ins do -- draw input neurons
            circfill(lindx*pad,r*pad, radius, graph_lib.heat_pal[1])
            circ(lindx*pad,r*pad, radius, graph_lib.heat_pal[5])
        end
        if lindx==#self.layers then
            -- draw output neurons
            for c=1,outs do -- draw input neurons
                circfill((lindx+1)*pad,c*pad, radius, graph_lib.heat_pal[3])
                circ((lindx+1)*pad,c*pad, radius, graph_lib.heat_pal[5])
            end
        end
    end
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
    local res=np_dot_vm(x, self.weights) -- X dot W
    res=np_add_vec(res, self.bias) -- add bias
    self:apply_activation(res) -- apply activiation fn
    return res
end
function layer:apply_activation(vec)
    if self.activation=="tanh" then
        np_vec_func(vec, np_tanh)
    else -- assume "sigmoid"
        np_vec_func(vec, np_sigmoid)
    end
end
fcnn={}
-- constructor for fully-connected neural network
-- input is array of ints. E.g. {3,5,3}
function fcnn:new(_layers)
    local this={}
    this.layers={}
    if (_layers != nil) then
        -- create all neurons
        for lindx=1,#_layers do -- each layer
            this.layers[lindx] = {}
            for nindx=1,_layers[lindx] do -- each neuron
                local n=node:new(lindx*24,nindx*24)
                add(this.layers[lindx], n)
            end
        end
        -- connect neurons
        for lindx,l in ipairs(this.layers) do -- each layer
            -- if next layer exists, connect
            if (lindx+1 <= #this.layers) then
                for nindx,n in ipairs(l) do -- each current layer node
                    for nl_indx,nl_n in ipairs(this.layers[lindx+1]) do -- each next layer node
                        n:connect(nl_n)
                    end
                end
            end
        end
    end


    self.__index=self
    setmetatable(this,self)
    return this
end
function fcnn:feedforward()
    for l in (self.layers) do -- each layer
        for n in l do -- each neuron
            for e in n.edges do -- each edge
                --
            end
        end
    end
end
function draw_nn(_net)
    --
    for n in all(_net.layers[1]) do
        draw_graph(n)
    end
end

-->8
-- num-pico lib
-- returns a n-by-m array with random values (0,1)
function np_rand_mat(_n,_m)
    local a={}
    for i=1,_n do -- each row
        a[i]={}
        for j=1,_m do -- each col
            a[i][j]=rnd()
        end
    end
    return a
end
-- returns a n-dim array with random values (0,1)
function np_rand_vec(_n)
    local a={}
    for i=1,_n do
        a[i]=rnd()
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
-- matrix component-wise addtion
-- a is m-by-n, b is n-by-p
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
-- applies function to each vector component
function np_vec_func(_v,_func)
    for i=1,#_v do
        _v[i]=_func(_v[i])
    end
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
-- hiberbolic tangent
function np_tanh(x)
	return np_sinh(x)/np_cosh(x)
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
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00700700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00077000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00077000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00700700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

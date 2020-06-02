pico-8 cartridge // http://www.pico-8.com
version 27
__lua__
-- pico learn
-- by blearn
function _init()
    draw=true
    job=nil
    rectfill(0,0,127,127,0)
    my_init_nn()
    job=cocreate(my_train_cor)

    if my_dat == nil then
        my_dat=0
    end
    test()
end
function _update60()
    if job and costatus(job) != 'dead' then
        coresume(job)
    else
        job=nil
    end
end
function _draw()
    my_draw_nn()
    print(my_dat,2,2,7)
end
-->8
-- demo
function test()
    printh("my_dat="..tostr(my_dat+1), "dat", true)
end
function my_init_nn()
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
    learing_rate=0.03
    --learing_rate=0.003
    epochs=10000
    num_samples=10
    num_test_samples=100

    -- generate random test data
    x = {}
    y = {}
    x_test = {}
    y_test = {}
    for s=1,num_samples do -- each train sample
        x[s]={}
        for i=1,num_inputs do -- each input
            x[s][i]=rnd(2)-1
        end
        y[s]={}
        for l=1,num_outputs do -- each label
            y[s][l]=1.0
            --y[s][l]=rnd(2)-1
        end
    end

    for s=1,num_test_samples do -- each test sample
        x_test[s]={}
        for i=1,num_inputs do -- each input
            x_test[s][i]=rnd(2)-1
        end
        y_test[s]={}
        for l=1,num_outputs do -- each label
            y_test[s][l]=1.0
            --y_test[s][l]=rnd(2)-1
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
-- coroutine for training nn
function my_train_cor()
    -- train
    while not(done_training) do
        if (net.epoch<epochs) then -- ready for next epoch
            prev_epoch=net.epoch
            net:train_step(x[prev_sample],y[prev_sample],learing_rate)
            add(errors, net:mse(y[1],x[1]))
            if (prev_sample<num_samples) then -- continue batch
                prev_sample+=1
            else -- batch complete
                prev_sample=1
                net:accuracy(x, y)
                net.epoch+=1
            end
        elseif (net.epoch>=epochs and not(done_training)) then -- training over
            done_training=true
            net:accuracy(x_test, y_test)
        end
        yield()
    end
end
function my_update_nn()
    --output=net:feedforward(np_rand_vec(num_inputs))

    -- train
    if (net.epoch<epochs) then -- ready for next epoch
        prev_epoch=net.epoch
        net:train_step(x[prev_sample],y[prev_sample],learing_rate)
        add(errors, net:mse(y[1],x[1]))
        if (prev_sample<num_samples) then -- continue batch
            prev_sample+=1
        else -- batch complete
            prev_sample=1
            net:accuracy(x, y)
            net.epoch+=1
        end
    elseif (net.epoch>=epochs and not(done_training)) then -- training over
        done_training=true
        net:accuracy(x_test, y_test)
    end
end
function my_draw_nn()
    if (draw) then
        rectfill(0,0,127,127,0)
        rect(0,0,127,127,6)
        net:draw_net()
        net:draw_stats()
        net:draw_index()
        --print("output: ", 16,64, 6)
        --for i=1,#output do
        --    print(tostr(output[i]), 32,72+(i*8), 6)
        --end
    end
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
-- graph lib
#include graph-lib.lua
-->8
-- learning lib
#include pico-learn.lua
-->8
-- num-pico lib
#include num-pico.lua

-->8
-- data
#include dat.p8l
__gfx__
00000000700303080000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000e0010c090000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00700700800c070a0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00077000200509030000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00077000100404010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00700700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

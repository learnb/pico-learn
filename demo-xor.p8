pico-8 cartridge // http://www.pico-8.com
version 27
__lua__
-- pico learn demo - xor
-- by blearn
function _init()
    draw=false
    job=nil
    --rectfill(0,0,127,127,0)
    my_init_nn()
    job=cocreate(my_train_cor)
end
function _update60()
    if job != nil then
        local status = costatus(job)
        --printh("status: "..status)
        if job and status != 'dead' then
            coresume(job)
        elseif status == 'dead' then
            printh("job status: "..status)
            printh(trace(job))
            job=nil

            printh("(0,0): "..net:predict({0,0}).." truth: "..np_argmax(y[1]))
            printh("(0,1): "..net:predict({0,1}).." truth: "..np_argmax(y[2]))
            printh("(1,0): "..net:predict({1,0}).." truth: "..np_argmax(y[3]))
            printh("(1,1): "..net:predict({1,1}).." truth: "..np_argmax(y[4]))
        end
    end
end
function _draw()
    rectfill(0,0,127,127,0)
    net:draw_stats()
end
-->8
-- demo
function my_init_nn()
    num_inputs=2
    num_outputs=2
    hidden_nodes=3
    net=nn:new()
    net:add_layer(layer:new(num_inputs,hidden_nodes,"tanh"))
    net:add_layer(layer:new(hidden_nodes,num_outputs,"sigmoid"))

    output={}
    output=net:feedforward(np_vec_rand(num_inputs))

    -- configure training
    learning_rate=0.1
    --learning_rate=0.003
    epochs=500
    num_samples=4
    num_test_samples=4

    -- generate train/test data
    x = {}
    y = {}

    -- x[sample_indx][node_indx]

    x[1] = {}
    x[1][1] = 0
    x[1][2] = 0
    x[2] = {}
    x[2][1] = 0
    x[2][2] = 1
    x[3] = {}
    x[3][1] = 1
    x[3][2] = 0
    x[4] = {}
    x[4][1] = 1
    x[4][2] = 1

    y[1] = {}
    y[1][1] = 1
    y[1][2] = -1
    y[2] = {}
    y[2][1] = -1
    y[2][2] = 1
    y[3] = {}
    y[3][1] = -1
    y[3][2] = 1
    y[4] = {}
    y[4][1] = 1
    y[4][2] = -1
 
    x_test = {}
    x_test[1] = {}
    x_test[1][1] = 0
    x_test[1][2] = 0
    x_test[2] = {}
    x_test[2][1] = 0
    x_test[2][2] = 0
    x_test[3] = {}
    x_test[3][1] = 0
    x_test[3][2] = 1
    x_test[4] = {}
    x_test[4][1] = 0
    x_test[4][2] = 1
    x_test[5] = {}
    x_test[5][1] = 1
    x_test[5][2] = 0
    x_test[6] = {}
    x_test[6][1] = 1
    x_test[6][2] = 1
    y_test = {}
    y_test[1] = {}
    y_test[1][1] = 1
    y_test[1][2] = -1
    y_test[2] = {}
    y_test[2][1] = 1
    y_test[2][2] = -1
    y_test[3] = {}
    y_test[3][1] = -1
    y_test[3][2] = 1
    y_test[4] = {}
    y_test[4][1] = -1
    y_test[4][2] = 1
    y_test[5] = {}
    y_test[5][1] = -1
    y_test[5][2] = 1
    y_test[6] = {}
    y_test[6][1] = 1
    y_test[6][2] = -1
    

    prev_sample=1
    prev_epoch=-1

    errors={}
    done_training=false
    -- train
    --net:train_step(x[1],y[1],learning_rate)
    --net:train(x,y,learning_rate,epochs)
    --net:accuracy(x_test, y_test)
end
-- coroutine for training nn
function my_train_cor()
    -- train
    while not(done_training) do
        if (net.epoch<epochs) then -- ready for next epoch
            prev_epoch=net.epoch
            net:train_step(x[prev_sample],y[prev_sample],learning_rate)
            add(errors, net:mse(y[1],x[1]))
            if (prev_sample<num_samples) then -- continue batch
                prev_sample+=1
            else -- batch complete
                prev_sample=1
                net:accuracy(x_test, y_test)
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
    --output=net:feedforward(np_vec_rand(num_inputs))

    -- train
    if (net.epoch<epochs) then -- ready for next epoch
        prev_epoch=net.epoch
        net:train_step(x[prev_sample],y[prev_sample],learning_rate)
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

-->8

-->8
-- graph lib
--#include graph-lib.lua
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

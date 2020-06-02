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
using LinearAlgebra
using DataStructures


struct Event:

    time::Float64
    i::Int64
    j::Int64
    event_type::String
    insert_time::Float64

end

function Base.:<(x::Event, y::Event)
    return x.time < y.time
end

struct Ensemble
    
    events = PriorityQueue{Int64,Event}()
    
end    

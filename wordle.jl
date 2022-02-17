println(pwd())
include("utils.jl")
using POMDPs
using POMDPModelTools
using SARSOP
using POMDPModels # this contains the TigerPOMDP model
using BeliefUpdaters
using POMDPLinter

cache_word_scores(NONSOLUTION_WORDS,SOLUTION_WORDS)
println("Cached word scores")

struct WordlePOMDP <: POMDP{String,String,UInt8}
    word_scores::Array{UInt8}
    all_scores::Array{UInt8}
    score_to_idx::Dict{UInt8,Int}
    solution_words::Vector{String}
    all_words::Vector{String}
    solution_word_to_index::Dict{String,Int}
    all_word_to_index::Dict{String,Int}

    num_solutions::Int
    num_guesses::Int
end

function WordlePOMDP()
    all_scores = collect(Set(WORD_SCORES))
    score_to_idx = Dict((score,i) for (i,score) in enumerate(all_scores))
    return WordlePOMDP(WORD_SCORES,
                        all_scores,
                        score_to_idx,
                        SOLUTION_WORDS,
                        ALL_WORDS,
                        SOLUTION_WORD_TO_IDX,
                        ALL_WORD_TO_IDX,
                        length(SOLUTION_WORDS),
                        length(ALL_WORDS))
end

function init_uniform(p::WordlePOMDP)
    n = p.num_solutions
    prob = ones(n+1)/n
    prob[end] = 0
    return DiscreteBelief(p,prob)
end

function POMDPs.initialize_belief(up::DiscreteUpdater{WordlePOMDP}, d)
    # return uniform_belief(up.p)
    return init_uniform(up.pomdp)
end

function POMDPs.update(up::DiscreteUpdater{WordlePOMDP}, b, a, o)
    n = up.pomdp.num_solutions
    current_inds = collect(1:n)[b.probs .!= 0]
    aid = up.pomdp.all_word_to_index[a]
    valid_inds = filter(i -> up.pomdp.word_scores[aid,i] == o,
                        current_inds)
    num_valid = length(valid_inds)
    prob = zeros(n)
    prob[valid_inds] = 1/num_valid
    return DiscreteBelief(up.pomdp,prob)
end

POMDPs.states(p::WordlePOMDP) = cat(p.solution_words,["-1"],dims=1)
POMDPs.actions(p::WordlePOMDP) = p.all_words
POMDPs.observations(p::WordlePOMDP) = p.all_scores
POMDPs.discount(p::WordlePOMDP) = 0.9
POMDPs.stateindex(p::WordlePOMDP, s) = (s == "-1") ? p.num_solutions + 1 : p.solution_word_to_index[s]
POMDPs.actionindex(p::WordlePOMDP, a) = p.all_word_to_index[a]
POMDPs.obsindex(p::WordlePOMDP, o) = p.score_to_idx[o]

function POMDPs.transition(p::WordlePOMDP, s, a)
    if a == s
        return Deterministic("-1")
    else
        return Deterministic(s)
    end
end
function POMDPs.observation(p::WordlePOMDP, a, s)
    if s == "-1" 
        return Deterministic(0)
    else  
        aid = p.all_word_to_index[a]
        sid = p.solution_word_to_index[s]
        return Deterministic(p.word_scores[aid,sid])
    end
end
function POMDPs.reward(p::WordlePOMDP,s,a)
    if s == "-1"
        return 0
    else
        return s == a ? 1 : 0
    end
end
POMDPs.initialstate(p::WordlePOMDP) = init_uniform(p)
POMDPs.isterminal(p::WordlePOMDP,s) = (s == "-1") ? true : false  
using POMDPXFiles
function POMDPXFiles.trans_xml(pomdp::WordlePOMDP, pomdpx::POMDPXFile, out_file::IOStream)
    print("Able to overwrite")
    pomdp_states = ordered_states(pomdp)
    pomdp_pstates = ordered_states(pomdp)
    acts = ordered_actions(pomdp)

    aname = pomdpx.action_name
    var = pomdpx.state_name

    write(out_file, "\t<StateTransitionFunction>\n")
    str = "\t\t<CondProb>\n"
    str = "$(str)\t\t\t<Var>$(var)1</Var>\n"
    str = "$(str)\t\t\t<Parent>$(aname) $(var)0</Parent>\n"
    str = "$(str)\t\t\t<Parameter>\n"
    write(out_file, str)
    for (i, s) in enumerate(pomdp_states)
        # print("$i,")
        if isterminal(pomdp, s) # if terminal, just remain in the same state
            str = "\t\t\t\t<Entry>\n"
            str = "$(str)\t\t\t\t\t<Instance>* s$(i-1) s$(i-1)</Instance>\n"
            str = "$(str)\t\t\t\t\t<ProbTable>1.0</ProbTable>\n"
            str = "$(str)\t\t\t\t</Entry>\n"
            write(out_file, str)
        else
            for (ai, a) in enumerate(acts)
                d = transition(pomdp, s, a)
                p = 1.0
                if s == a
                    j = length(pomdp_pstates) 
                else
                    j = i 
                end
                p = 1.0
                
                # for (j, sp) in enumerate(pomdp_pstates)
                #     p = pdf(d, sp)
                #     if p > 0.0
                str = "\t\t\t\t<Entry>\n"
                str = "$(str)\t\t\t\t\t<Instance>a$(ai-1) s$(i-1) s$(j-1)</Instance>\n"
                str = "$(str)\t\t\t\t\t<ProbTable>$(p)</ProbTable>\n"
                str = "$(str)\t\t\t\t</Entry>\n"
                write(out_file, str)
                #     end
                # end
            end
        end
    end
    str = "\t\t\t</Parameter>\n"
    str = "$(str)\t\t</CondProb>\n"
    write(out_file, str)
    write(out_file, "\t</StateTransitionFunction>\n\n\n")
    println("Done with trans_xml")
    return nothing
end

pomdp = WordlePOMDP()
solver = SARSOPSolver()

policy = solve(solver, pomdp)
println("Done")

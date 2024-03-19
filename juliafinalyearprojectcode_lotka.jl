using Flux, DiffEqFlux, DifferentialEquations, Plots

datasize = 75

function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end
u0 = [4, 4]
tbegin = 0.0
tend = 8
t = range(tbegin, tend, length=datasize)
tspan = (tbegin, tend)
trange = range(tbegin, tend, length = datasize)
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(lotka_volterra,u0,tspan,p)
dataset_ts = Array(solve(prob, Tsit5(), saveat=trange))



u1 = [3, 2]
tbegin = 0.0
tend = 8
t = range(tbegin, tend, length=datasize)
tspan = (tbegin, tend)
trange = range(tbegin, tend, length = datasize)
p = [1.5,1.0,3.0,1.0]
prob2 = ODEProblem(lotka_volterra,u1,tspan,p)
dataset_ts2 = Array(solve(prob2, Tsit5(), saveat=trange))


u2 = [6, 2]
tbegin = 0.0
tend = 8
t = range(tbegin, tend, length=datasize)
tspan = (tbegin, tend)
trange = range(tbegin, tend, length = datasize)
p = [1.5,1.0,3.0,1.0]
prob3 = ODEProblem(lotka_volterra,u2,tspan,p)
dataset_ts3 = Array(solve(prob3, Tsit5(), saveat=trange))

u3 = [6, 6]
tbegin = 0.0
tend = 8
t = range(tbegin, tend, length=datasize)
tspan = (tbegin, tend)
trange = range(tbegin, tend, length = datasize)
p = [1.5,1.0,3.0,1.0]
prob4 = ODEProblem(lotka_volterra,u3,tspan,p)
dataset_ts4 = Array(solve(prob4, Tsit5(), saveat=trange))


u4 = [3, 8]
tbegin = 0.0
tend = 8
t = range(tbegin, tend, length=datasize)
tspan = (tbegin, tend)
trange = range(tbegin, tend, length = datasize)
p = [1.5,1.0,3.0,1.0]
prob5 = ODEProblem(lotka_volterra,u4,tspan,p)
dataset_ts5 = Array(solve(prob5, Tsit5(), saveat=trange))

u5 = [5, 8]
tbegin = 0.0
tend = 8
t = range(tbegin, tend, length=datasize)
tspan = (tbegin, tend)
trange = range(tbegin, tend, length = datasize)
p = [1.5,1.0,3.0,1.0]
prob6 = ODEProblem(lotka_volterra,u5,tspan,p)
dataset_ts6 = Array(solve(prob6, Tsit5(), saveat=trange))



dudt = Chain(Dense(2, 50, tanh),
             Dense(50, 2))

reltol = 1e-7
abstol = 1e-9
n_ode = NeuralODE(dudt, tspan, Tsit5(), saveat=trange, reltol=reltol,abstol=abstol)
ps = Flux.params(n_ode.p)

function loss_n_ode()
  pred = n_ode(u0)
  pred2 = n_ode(u1)
  pred3 = n_ode(u2)
  loss = sum(abs2, dataset_ts[1,:] .- pred[1,:]) +
         sum(abs2, dataset_ts[2,:] .- pred[2,:])
  loss
end

n_epochs = 26000
learning_rate = 0.01
data = Iterators.repeated((), n_epochs)
opt = ADAM(learning_rate)

cb = function ()
  loss = loss_n_ode()
  println("Loss: ", loss)
end

println();
cb()

Flux.train!(loss_n_ode, ps, data, opt, cb=cb)



u_array = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8]]

           
datasets = []
datasetspred = []
totaln = 0

@time for j in u_array
  datasetpred_loop2 = n_ode(j)
  push!(datasetspred, datasetpred_loop2)
end

@time for u0i in u_array
  probtest = ODEProblem(lotka_volterra, u0i, tspan, p)
  dataset_loop = Array(solve(probtest, Tsit5(), saveat = trange))
  push!(datasets, dataset_loop)
end




arraypoints = []


for m in u_array
  currentLoss = sum(abs, datasets[findfirst(x->x==m, u_array)][1,:]-datasetspred[findfirst(x->x==m, u_array)][1,:]) +
  sum(abs, datasets[findfirst(x->x==m, u_array)][2,:] - datasetspred[findfirst(x->x==m, u_array)][2,:])
  push!(arraypoints, [m[1], m[2], currentLoss])
  global totaln += currentLoss
end

print(totaln)




for j in u_array
  pl2 = plot(
  trange,
  datasets[findfirst(x->x==j, u_array)][1,:],
  linewidth=2, ls=:dash,
  title="Neural ODE  $j",
  xaxis="t",
  label="original timeseries x(t)",
  legend=:right)

  pl2 = plot!(
    trange,
    datasets[findfirst(x->x==j, u_array)][2,:],
    linewidth=2, ls=:dash,
    label="original timeseries y(t)")
  display(pl2)

  pl2 = plot!(
  trange,
  datasetspred[findfirst(x->x==j, u_array)][1,:],
  linewidth=1,
  label="predicted timeseries x(t)")

  pl2 = plot!(
    trange,
    datasetspred[findfirst(x->x==j, u_array)][2,:],
    linewidth=1,
    label="predicted timeseries y(t)")
  display(pl2)


  pl3 = plot(datasets[findfirst(x->x==j, u_array)][1,:], datasets[findfirst(x->x==j, u_array)][2,:], xlabel="x(t)", ylabel="y(t)", label="original time series", title="Predator and Prey Phase Portrait $j", color="red")
  pl3= plot!(datasetspred[findfirst(x->x==j, u_array)][1,:], datasetspred[findfirst(x->x==j, u_array)][2,:], label = "predicted time series", color="blue")
  display(pl3)
end



x = [p[1] for p in arraypoints]
y = [p[2] for p in arraypoints]
z = [p[3] for p in arraypoints]
  
scatter(x,y,z, xlabel="x", ylabel="y", zlabel="abs difference")

x2 = [4]
y2 = [4]
z2 = [0]

scatter!(x2,y2,z2, color="red", marker=:x)
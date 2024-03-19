using Flux, DiffEqFlux, DifferentialEquations, Plots

datasize = 75

function lorenz(du, u, p, t)
  x, y, z = u
  σ, ρ, β = p
  du[1] = σ * (y - x)
  du[2] = x * (ρ - z) - y
  du[3] = x * y - β * z
end
u0 = [1, 1, 1]
tbegin = 0.0
tend = 1.5
t = range(tbegin, tend, length=datasize)
tspan = (tbegin, tend)
trange = range(tbegin, tend, length = datasize)
p = [10.0, 28.0, 8/3]
prob = ODEProblem(lorenz,u0,tspan,p)
dataset_ts = Array(solve(prob, Tsit5(), saveat=trange))



u1 = [1.5, 0.5, 2]
tbegin = 0.0
tend = 1.5
t = range(tbegin, tend, length=datasize)
tspan = (tbegin, tend)
trange = range(tbegin, tend, length = datasize)
prob2 = ODEProblem(lorenz,u1,tspan,p)
dataset_ts2 = Array(solve(prob2, Tsit5(), saveat=trange))


u2 = [1, 3, 4]
tbegin = 0.0
tend = 1.5
t = range(tbegin, tend, length=datasize)
tspan = (tbegin, tend)
trange = range(tbegin, tend, length = datasize)
prob3 = ODEProblem(lorenz,u2,tspan,p)
dataset_ts3 = Array(solve(prob3, Tsit5(), saveat=trange))

u3 = [2, 3, 1]
tbegin = 0.0
tend = 1.5
t = range(tbegin, tend, length=datasize)
tspan = (tbegin, tend)
trange = range(tbegin, tend, length = datasize)
prob4 = ODEProblem(lorenz,u3,tspan,p)
dataset_ts4 = Array(solve(prob4, Tsit5(), saveat=trange))


u4 = [1,4,2]
tbegin = 0.0
tend = 1.5
t = range(tbegin, tend, length=datasize)
tspan = (tbegin, tend)
trange = range(tbegin, tend, length = datasize)
prob5 = ODEProblem(lorenz,u4,tspan,p)
dataset_ts5 = Array(solve(prob5, Tsit5(), saveat=trange))

u5 = [3, 3, 3]
tbegin = 0.0
tend = 1.5
t = range(tbegin, tend, length=datasize)
tspan = (tbegin, tend)
trange = range(tbegin, tend, length = datasize)
prob6 = ODEProblem(lorenz,u5,tspan,p)
dataset_ts6 = Array(solve(prob6, Tsit5(), saveat=trange))





#sol = solve(prob)
#using Plots
#plot(sol)

dudt = Chain(Dense(3, 256, relu),
             Dense(256, 3))

reltol = 1e-7
abstol = 1e-9
n_ode = NeuralODE(dudt, tspan, Tsit5(), saveat=trange, reltol=reltol,abstol=abstol)
ps = Flux.params(n_ode.p)

function loss_n_ode()
  pred = n_ode(u0)
  pred2 = n_ode(u1)
  pred3 = n_ode(u2)
  pred4 = n_ode(u3)
  pred5 = n_ode(u4)
  pred6 = n_ode(u5)
  loss = sum(abs2, dataset_ts[1,:] .- pred[1,:]) +
  sum(abs2, dataset_ts[2,:] .- pred[2,:]) + sum(abs2, dataset_ts[3,:] .- pred[3,:])
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

using BSON: @save
@save "mymod12el.bson" ps

using BSON: @save
@save "mymodelno31de.bson" n_ode



# Define an array of 64 initial conditions
u_array = [ [0.5, 0.5, 0.5], [0.5, 0.5, 1], [0.5, 0.5, 1.5], [0.5, 0.5, 2], [0.5, 1, 0.5], [0.5, 1, 1], [0.5, 1, 1.5], [0.5, 1, 2], [0.5, 1.5, 0.5], [0.5, 1.5, 1], [0.5, 1.5, 1.5], [0.5, 1.5, 2], [0.5, 2, 0.5], [0.5, 2, 1], [0.5, 2, 1.5], [0.5, 2, 2], [1, 0.5, 0.5], [1, 0.5, 1], [1, 0.5, 1.5], [1, 0.5, 2], [1, 1, 0.5], [1, 1, 1], [1, 1, 1.5], [1, 1, 2], [1, 1.5, 0.5], [1, 1.5, 1], [1, 1.5, 1.5], [1, 1.5, 2], [1, 2, 0.5], [1, 2, 1], [1, 2, 1.5], [1, 2, 2], [1.5, 0.5, 0.5], [1.5, 0.5, 1], [1.5, 0.5, 1.5], [1.5, 0.5, 2], [1.5, 1, 0.5], [1.5, 1, 1], [1.5, 1, 1.5], [1.5, 1, 2], [1.5, 1.5, 0.5], [1.5, 1.5, 1], [1.5, 1.5, 1.5], [1.5, 1.5, 2], [1.5, 2, 0.5], [1.5, 2, 1], [1.5, 2, 1.5], [1.5, 2, 2], [2, 0.5, 0.5], [2, 0.5, 1], [2, 0.5, 1.5], [2, 0.5, 2], [2, 1, 0.5], [2, 1, 1], [2, 1, 1.5], [2, 1, 2], [2, 1.5, 0.5], [2, 1.5, 1], [2, 1.5, 1.5], [2, 1.5, 2], [2, 2, 0.5], [2, 2, 1], [2, 2, 1.5], [2, 2, 2] ]

# Solve the ODE for each initial condition and store the results in an array
datasets = []
datasetspred = []
@time for u0i in u_array
    probtest = ODEProblem(lorenz, u0i, tspan, p)
    dataset_loop = Array(solve(probtest, Tsit5(), saveat = trange))
    push!(datasets, dataset_loop)
end

@time for j in u_array
  datasetpred_loop2 = n_ode(j)
  push!(datasetspred, datasetpred_loop2)
end

arraypointsxy = []
arraypointsxz = []
arraypointsyz = []
totaln = 0

for m in u_array
  currentLoss = sum(abs, datasets[findfirst(x->x==m, u_array)][1,:]-datasetspred[findfirst(x->x==m, u_array)][1,:]) +
  sum(abs, datasets[findfirst(x->x==m, u_array)][2,:] - datasetspred[findfirst(x->x==m, u_array)][2,:]) +
  sum(abs, datasets[findfirst(x->x==m, u_array)][3,:] - datasetspred[findfirst(x->x==m, u_array)][3,:])
  push!(arraypointsxy, [m[1], m[2], currentLoss])
  push!(arraypointsxz, [m[1], m[3], currentLoss])
  push!(arraypointsyz, [m[2], m[3], currentLoss])
  global totaln += currentLoss
end

print(totaln)


for j in u_array
  pl2 = plot(
  trange,
  datasets[findfirst(x->x==j, u_array)][1,:],
  linewidth=2, ls=:dash,
  title="Neural ODE for forecasting  $j",
  xaxis="t",
  label="original timeseries x(t)",
  legend=:right)

  pl2 = plot!(
    trange,
    datasets[findfirst(x->x==j, u_array)][2,:],
    linewidth=2, ls=:dash,
    label="original timeseries y(t)")

  pl2 = plot!(
    trange,
    datasets[findfirst(x->x==j, u_array)][3,:],
    linewidth=2, ls=:dash,
    label="original timeseries z(t)"
    
  )
    
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

    pl2 = plot!(
        trange,
        datasetspred[findfirst(x->x==j, u_array)][3,:],
        linewidth=1,
        label="predicted timeseries z(t)")
  display(pl2)
end

xt = [1]
yt = [1]
zt = [0]
  

x1 = [p[1] for p in arraypointsxy]
y1 = [p[2] for p in arraypointsxy]
differenceloss = [p[3] for p in arraypointsxy]
  
scatter(x1,y1,differenceloss, xlabel="x", ylabel="y", zlabel="abs difference")
scatter!(xt,yt,zt, color="red", marker=:x)

display(plot!())

plot()

xt2 = [1]
yt2 = [1]
zt2 = [0]

x2 = [p[1] for p in arraypointsxz]
z2 = [p[2] for p in arraypointsxz]
differenceloss2 = [p[3] for p in arraypointsxz]

scatter(x2,z2,differenceloss2, xlabel="x", ylabel="z", zlabel="abs difference")
scatter!(xt2,yt2,zt2, color="red", marker=:x)

display(plot!())
plot()

xt3 = [1]
yt3 = [1]
zt3 = [0]

y3 = [p[1] for p in arraypointsyz]
z3 = [p[2] for p in arraypointsyz]
differenceloss3 = [p[3] for p in arraypointsyz]

scatter(y3,z3,differenceloss3, xlabel="y", ylabel="z", zlabel="abs difference")
scatter!(xt3,yt3,zt3, color="red", marker=:x)
display(plot!())
plot()
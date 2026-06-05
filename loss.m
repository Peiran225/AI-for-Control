{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 Menlo-Regular;}
{\colortbl;\red255\green255\blue255;\red8\green0\blue255;\red0\green0\blue0;\red15\green112\blue16;
\red148\green0\blue242;}
{\*\expandedcolortbl;;\cssrgb\c5490\c0\c100000;\cssrgb\c0\c0\c0;\cssrgb\c0\c50196\c7451;
\cssrgb\c65490\c3529\c96078;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs20 \cf2 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 function \cf0 \strokec3 d = loss(u_vec, T, n, m, beta)\
M = diag(0.5 * ones(m, 1));\
x = linspace(0, 1, m)'; \
r = (2 ./ (1 + 3 * x.^4));\
R = diag(r);\
Phi1 = diag(1./(1 + x.^2));\
\
[N, G_N] = simulateTumorGrowth(u_vec, T, n, m);\
\
s = beta * (R - G_N * M) * N./ (beta * Phi1 * N);\
size(s)\
size(N)\
\
d = s - u_vec\
\cf2 \strokec2 end\cf0 \strokec3 \
\
\
\cf2 \strokec2 function \cf0 \strokec3 [N, G_N] = simulateTumorGrowth(u_vec, T, n, m)\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4 %simulateTumorGrowth Simulates tumor dynamics for a single drug without mutation.\cf0 \strokec3 \
\cf4 \strokec4 %   N = simulateTumorGrowth(U_VEC) calculates the tumor population\cf0 \strokec3 \
\cf4 \strokec4 %   matrix N based on a single control input schedule provided in U_VEC.\cf0 \strokec3 \
\cf4 \strokec4 %\cf0 \strokec3 \
\cf4 \strokec4 %   INPUT:\cf0 \strokec3 \
\cf4 \strokec4 %   u_vec   - A numerical vector representing the drug schedule.\cf0 \strokec3 \
\cf4 \strokec4 %             The vector must have a length of n+1 (default is 101).\cf0 \strokec3 \
\cf4 \strokec4 %\cf0 \strokec3 \
\cf4 \strokec4 %   OUTPUT:\cf0 \strokec3 \
\cf4 \strokec4 %   N       - The resulting m x (n+1) tumor population matrix, where m is\cf0 \strokec3 \
\cf4 \strokec4 %             the number of traits and n is the number of time divisions.\cf0 \strokec3 \
\
\cf4 \strokec4 %% 1. Define All Necessary Simulation Parameters\cf0 \strokec3 \
\cf4 \strokec4 % Time and Trait Discretization\cf0 \strokec3 \
\cf4 \strokec4 % T = 10; % Final time\cf0 \strokec3 \
\cf4 \strokec4 % n = 4; % Number of time divisions default: 100\cf0 \strokec3 \
\cf4 \strokec4 % m = 21; % Number of trait divisions\cf0 \strokec3 \
time = linspace(0, T, n + 1);\
\
\cf4 \strokec4 % Model Parameters\cf0 \strokec3 \
x = linspace(0, 1, m)'; \cf4 \strokec4 % Trait values\cf0 \strokec3 \
r = (2 ./ (1 + 3 * x.^4)); \cf4 \strokec4 % Replication rate \cf0 \strokec3 \
M = diag(0.5 * ones(m, 1)); \cf4 \strokec4 % Natural death rate \cf0 \strokec3 \
\
\cf4 \strokec4 % Drug Efficacy Profile (C in 30)\cf0 \strokec3 \
Phi1 = diag(1./(1 + x.^2));\
\cf4 \strokec4 %Phi1 = diag(1 + cos(0.25*pi * x).^2 .* ones(m, 1));\cf0 \strokec3 \
\
\cf4 \strokec4 % --- Construct the Growth Matrix R (NO MUTATION) ---\cf0 \strokec3 \
R = diag(r);\
\
\cf4 \strokec4 % Initial Conditions\cf0 \strokec3 \
n0 = 10 * ones(m, 1); \cf4 \strokec4 % Initial cell density\cf0 \strokec3 \
\
\cf4 \strokec4 % Package all parameters into a struct for convenience\cf0 \strokec3 \
param = struct(\cf5 \strokec5 'R'\cf0 \strokec3 , R, \cf5 \strokec5 'Phi1'\cf0 \strokec3 , Phi1, \cf5 \strokec5 'M'\cf0 \strokec3 , M, \cf5 \strokec5 'time'\cf0 \strokec3 , time, \cf5 \strokec5 'e'\cf0 \strokec3 , ones(m, 1) / m);\
\
\cf4 \strokec4 %% 2. Process the Control Input (u)\cf0 \strokec3 \
\cf4 \strokec4 % Validate input vector size\cf0 \strokec3 \
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2 if \cf0 \strokec3 (length(u_vec) ~= n + 1)\
    error(\cf5 \strokec5 'Control vector must have length n+1 (%d).'\cf0 \strokec3 , n + 1);\
\cf2 \strokec2 end\cf0 \strokec3 \
fprintf(\cf5 \strokec5 'Simulating with 1 drug (no mutation)...\\n'\cf0 \strokec3 );\
\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4 %% 3. Generate the Population Matrix (N)\cf0 \strokec3 \
\cf4 \strokec4 % Convert the control vector into a continuous function\cf0 \strokec3 \
u1_func = @(t) interp1(time, u_vec, t, \cf5 \strokec5 'pchip'\cf0 \strokec3 );\
\
\cf4 \strokec4 % Solve the system of ODEs using the control function\cf0 \strokec3 \
[N, G_N] = eulerMethod(u1_func, param, n0);\
fprintf(\cf5 \strokec5 'Simulation complete.\\n'\cf0 \strokec3 );\
\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2 end \cf4 \strokec4 % End of main function\cf0 \strokec3 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4 %% 4. Local Function Definitions\cf0 \strokec3 \
\cf4 \strokec4 % All necessary helper functions are included below.\cf0 \strokec3 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2 function \cf0 \strokec3 [N, G_N] = eulerMethod(u1, param, n0)\
    dt = param.time(2) - param.time(1);\
    N = zeros(length(n0), length(param.time));\
    N(:, 1) = n0;\
    \cf4 \strokec4 % Forward integration for N\cf0 \strokec3 \
    \cf2 \strokec2 for \cf0 \strokec3 k = 1:length(param.time) - 1\
        t = param.time(k);\
        N_current = N(:, k);\
        [dNdt, G_N] = dynamics(t, N_current, u1, param);\
        N(:, k + 1) = N_current + dt * dNdt;\
    \cf2 \strokec2 end\cf0 \strokec3 \
\cf2 \strokec2 end\cf0 \strokec3 \
\
\cf2 \strokec2 function \cf0 \strokec3 [dNdt, G_N] = dynamics(t, N, u1, param)\
    N = N(:);\
    N_total = param.e' * N;\
    G_N = log(1 + N_total);\
    \
    \cf4 \strokec4 % Dynamics equation for a single drug a\cf0 \strokec3 \
    dNdt = (param.R - param.Phi1 * u1(t) - param.M * G_N) * N;\
\cf2 \strokec2 end\cf0 \strokec3 \
\
\
\
\
}
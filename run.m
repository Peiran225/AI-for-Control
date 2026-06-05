{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 Menlo-Regular;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red15\green112\blue16;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;\cssrgb\c0\c50196\c7451;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs20 \cf0 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 u_vec = rand(1, 5); \
T = 10; \cf3 \strokec3 % Final time\cf0 \strokec2 \
n = 4; \cf3 \strokec3 % Number of time divisions default: 100\cf0 \strokec2 \
m = 21; \cf3 \strokec3 % Number of trait divisions\cf0 \strokec2 \
beta = 0.3;\
    \
\
\pard\pardeftab720\partightenfactor0
\cf3 \strokec3 % Call your function\cf0 \strokec2 \
[N_results, G_N] = simulateTumorGrowth(u_vec);\
\
d = loss(u_vec, T, n, m, beta);\
\
display(d)\
}
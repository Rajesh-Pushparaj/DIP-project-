??

??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
z
hidden_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namehidden_1/kernel
s
#hidden_1/kernel/Read/ReadVariableOpReadVariableOphidden_1/kernel*
_output_shapes

:@*
dtype0
r
hidden_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namehidden_1/bias
k
!hidden_1/bias/Read/ReadVariableOpReadVariableOphidden_1/bias*
_output_shapes
:@*
dtype0
z
hidden_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ * 
shared_namehidden_2/kernel
s
#hidden_2/kernel/Read/ReadVariableOpReadVariableOphidden_2/kernel*
_output_shapes

:@ *
dtype0
r
hidden_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namehidden_2/bias
k
!hidden_2/bias/Read/ReadVariableOpReadVariableOphidden_2/bias*
_output_shapes
: *
dtype0
z
hidden_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @* 
shared_namehidden_3/kernel
s
#hidden_3/kernel/Read/ReadVariableOpReadVariableOphidden_3/kernel*
_output_shapes

: @*
dtype0
r
hidden_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namehidden_3/bias
k
!hidden_3/bias/Read/ReadVariableOpReadVariableOphidden_3/bias*
_output_shapes
:@*
dtype0
{
hidden_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?* 
shared_namehidden_4/kernel
t
#hidden_4/kernel/Read/ReadVariableOpReadVariableOphidden_4/kernel*
_output_shapes
:	@?*
dtype0
s
hidden_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namehidden_4/bias
l
!hidden_4/bias/Read/ReadVariableOpReadVariableOphidden_4/bias*
_output_shapes	
:?*
dtype0
|
hidden_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namehidden_5/kernel
u
#hidden_5/kernel/Read/ReadVariableOpReadVariableOphidden_5/kernel* 
_output_shapes
:
??*
dtype0
s
hidden_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namehidden_5/bias
l
!hidden_5/bias/Read/ReadVariableOpReadVariableOphidden_5/bias*
_output_shapes	
:?*
dtype0
{
hidden_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@* 
shared_namehidden_6/kernel
t
#hidden_6/kernel/Read/ReadVariableOpReadVariableOphidden_6/kernel*
_output_shapes
:	?@*
dtype0
r
hidden_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namehidden_6/bias
k
!hidden_6/bias/Read/ReadVariableOpReadVariableOphidden_6/bias*
_output_shapes
:@*
dtype0
v
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

:@*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/hidden_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/hidden_1/kernel/m
?
*Adam/hidden_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_1/kernel/m*
_output_shapes

:@*
dtype0
?
Adam/hidden_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/hidden_1/bias/m
y
(Adam/hidden_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_1/bias/m*
_output_shapes
:@*
dtype0
?
Adam/hidden_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/hidden_2/kernel/m
?
*Adam/hidden_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_2/kernel/m*
_output_shapes

:@ *
dtype0
?
Adam/hidden_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/hidden_2/bias/m
y
(Adam/hidden_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_2/bias/m*
_output_shapes
: *
dtype0
?
Adam/hidden_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*'
shared_nameAdam/hidden_3/kernel/m
?
*Adam/hidden_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_3/kernel/m*
_output_shapes

: @*
dtype0
?
Adam/hidden_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/hidden_3/bias/m
y
(Adam/hidden_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_3/bias/m*
_output_shapes
:@*
dtype0
?
Adam/hidden_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*'
shared_nameAdam/hidden_4/kernel/m
?
*Adam/hidden_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_4/kernel/m*
_output_shapes
:	@?*
dtype0
?
Adam/hidden_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/hidden_4/bias/m
z
(Adam/hidden_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_4/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/hidden_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/hidden_5/kernel/m
?
*Adam/hidden_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_5/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/hidden_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/hidden_5/bias/m
z
(Adam/hidden_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_5/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/hidden_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*'
shared_nameAdam/hidden_6/kernel/m
?
*Adam/hidden_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_6/kernel/m*
_output_shapes
:	?@*
dtype0
?
Adam/hidden_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/hidden_6/bias/m
y
(Adam/hidden_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_6/bias/m*
_output_shapes
:@*
dtype0
?
Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*%
shared_nameAdam/output/kernel/m
}
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
_output_shapes

:@*
dtype0
|
Adam/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/m
u
&Adam/output/bias/m/Read/ReadVariableOpReadVariableOpAdam/output/bias/m*
_output_shapes
:*
dtype0
?
Adam/hidden_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/hidden_1/kernel/v
?
*Adam/hidden_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_1/kernel/v*
_output_shapes

:@*
dtype0
?
Adam/hidden_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/hidden_1/bias/v
y
(Adam/hidden_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_1/bias/v*
_output_shapes
:@*
dtype0
?
Adam/hidden_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/hidden_2/kernel/v
?
*Adam/hidden_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_2/kernel/v*
_output_shapes

:@ *
dtype0
?
Adam/hidden_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/hidden_2/bias/v
y
(Adam/hidden_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_2/bias/v*
_output_shapes
: *
dtype0
?
Adam/hidden_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*'
shared_nameAdam/hidden_3/kernel/v
?
*Adam/hidden_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_3/kernel/v*
_output_shapes

: @*
dtype0
?
Adam/hidden_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/hidden_3/bias/v
y
(Adam/hidden_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_3/bias/v*
_output_shapes
:@*
dtype0
?
Adam/hidden_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*'
shared_nameAdam/hidden_4/kernel/v
?
*Adam/hidden_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_4/kernel/v*
_output_shapes
:	@?*
dtype0
?
Adam/hidden_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/hidden_4/bias/v
z
(Adam/hidden_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_4/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/hidden_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/hidden_5/kernel/v
?
*Adam/hidden_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_5/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/hidden_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/hidden_5/bias/v
z
(Adam/hidden_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_5/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/hidden_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*'
shared_nameAdam/hidden_6/kernel/v
?
*Adam/hidden_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_6/kernel/v*
_output_shapes
:	?@*
dtype0
?
Adam/hidden_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/hidden_6/bias/v
y
(Adam/hidden_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_6/bias/v*
_output_shapes
:@*
dtype0
?
Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*%
shared_nameAdam/output/kernel/v
}
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes

:@*
dtype0
|
Adam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/v
u
&Adam/output/bias/v/Read/ReadVariableOpReadVariableOpAdam/output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?F
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?E
value?EB?E B?E
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
	optimizer
	trainable_variables

	variables
regularization_losses
	keras_api

signatures
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

 kernel
!bias
"trainable_variables
#	variables
$regularization_losses
%	keras_api
h

&kernel
'bias
(trainable_variables
)	variables
*regularization_losses
+	keras_api
h

,kernel
-bias
.trainable_variables
/	variables
0regularization_losses
1	keras_api
h

2kernel
3bias
4trainable_variables
5	variables
6regularization_losses
7	keras_api
?
8iter

9beta_1

:beta_2
	;decay
<learning_ratemjmkmlmmmnmo mp!mq&mr'ms,mt-mu2mv3mwvxvyvzv{v|v} v~!v&v?'v?,v?-v?2v?3v?
f
0
1
2
3
4
5
 6
!7
&8
'9
,10
-11
212
313
f
0
1
2
3
4
5
 6
!7
&8
'9
,10
-11
212
313
 
?
	trainable_variables

=layers

	variables
regularization_losses
>layer_regularization_losses
?non_trainable_variables
@metrics
Alayer_metrics
 
[Y
VARIABLE_VALUEhidden_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables

Blayers
	variables
regularization_losses
Clayer_regularization_losses
Dnon_trainable_variables
Emetrics
Flayer_metrics
[Y
VARIABLE_VALUEhidden_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables

Glayers
	variables
regularization_losses
Hlayer_regularization_losses
Inon_trainable_variables
Jmetrics
Klayer_metrics
[Y
VARIABLE_VALUEhidden_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables

Llayers
	variables
regularization_losses
Mlayer_regularization_losses
Nnon_trainable_variables
Ometrics
Player_metrics
[Y
VARIABLE_VALUEhidden_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
?
"trainable_variables

Qlayers
#	variables
$regularization_losses
Rlayer_regularization_losses
Snon_trainable_variables
Tmetrics
Ulayer_metrics
[Y
VARIABLE_VALUEhidden_5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1

&0
'1
 
?
(trainable_variables

Vlayers
)	variables
*regularization_losses
Wlayer_regularization_losses
Xnon_trainable_variables
Ymetrics
Zlayer_metrics
[Y
VARIABLE_VALUEhidden_6/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_6/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1

,0
-1
 
?
.trainable_variables

[layers
/	variables
0regularization_losses
\layer_regularization_losses
]non_trainable_variables
^metrics
_layer_metrics
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

20
31

20
31
 
?
4trainable_variables

`layers
5	variables
6regularization_losses
alayer_regularization_losses
bnon_trainable_variables
cmetrics
dlayer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
1
0
1
2
3
4
5
6
 
 

e0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	ftotal
	gcount
h	variables
i	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

f0
g1

h	variables
~|
VARIABLE_VALUEAdam/hidden_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_5/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_5/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_6/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_6/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_5/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_5/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_6/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_6/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
x
serving_default_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputhidden_1/kernelhidden_1/biashidden_2/kernelhidden_2/biashidden_3/kernelhidden_3/biashidden_4/kernelhidden_4/biashidden_5/kernelhidden_5/biashidden_6/kernelhidden_6/biasoutput/kerneloutput/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_6583917
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#hidden_1/kernel/Read/ReadVariableOp!hidden_1/bias/Read/ReadVariableOp#hidden_2/kernel/Read/ReadVariableOp!hidden_2/bias/Read/ReadVariableOp#hidden_3/kernel/Read/ReadVariableOp!hidden_3/bias/Read/ReadVariableOp#hidden_4/kernel/Read/ReadVariableOp!hidden_4/bias/Read/ReadVariableOp#hidden_5/kernel/Read/ReadVariableOp!hidden_5/bias/Read/ReadVariableOp#hidden_6/kernel/Read/ReadVariableOp!hidden_6/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/hidden_1/kernel/m/Read/ReadVariableOp(Adam/hidden_1/bias/m/Read/ReadVariableOp*Adam/hidden_2/kernel/m/Read/ReadVariableOp(Adam/hidden_2/bias/m/Read/ReadVariableOp*Adam/hidden_3/kernel/m/Read/ReadVariableOp(Adam/hidden_3/bias/m/Read/ReadVariableOp*Adam/hidden_4/kernel/m/Read/ReadVariableOp(Adam/hidden_4/bias/m/Read/ReadVariableOp*Adam/hidden_5/kernel/m/Read/ReadVariableOp(Adam/hidden_5/bias/m/Read/ReadVariableOp*Adam/hidden_6/kernel/m/Read/ReadVariableOp(Adam/hidden_6/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp*Adam/hidden_1/kernel/v/Read/ReadVariableOp(Adam/hidden_1/bias/v/Read/ReadVariableOp*Adam/hidden_2/kernel/v/Read/ReadVariableOp(Adam/hidden_2/bias/v/Read/ReadVariableOp*Adam/hidden_3/kernel/v/Read/ReadVariableOp(Adam/hidden_3/bias/v/Read/ReadVariableOp*Adam/hidden_4/kernel/v/Read/ReadVariableOp(Adam/hidden_4/bias/v/Read/ReadVariableOp*Adam/hidden_5/kernel/v/Read/ReadVariableOp(Adam/hidden_5/bias/v/Read/ReadVariableOp*Adam/hidden_6/kernel/v/Read/ReadVariableOp(Adam/hidden_6/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*>
Tin7
523	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_6584396
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehidden_1/kernelhidden_1/biashidden_2/kernelhidden_2/biashidden_3/kernelhidden_3/biashidden_4/kernelhidden_4/biashidden_5/kernelhidden_5/biashidden_6/kernelhidden_6/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/hidden_1/kernel/mAdam/hidden_1/bias/mAdam/hidden_2/kernel/mAdam/hidden_2/bias/mAdam/hidden_3/kernel/mAdam/hidden_3/bias/mAdam/hidden_4/kernel/mAdam/hidden_4/bias/mAdam/hidden_5/kernel/mAdam/hidden_5/bias/mAdam/hidden_6/kernel/mAdam/hidden_6/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/hidden_1/kernel/vAdam/hidden_1/bias/vAdam/hidden_2/kernel/vAdam/hidden_2/bias/vAdam/hidden_3/kernel/vAdam/hidden_3/bias/vAdam/hidden_4/kernel/vAdam/hidden_4/bias/vAdam/hidden_5/kernel/vAdam/hidden_5/bias/vAdam/hidden_6/kernel/vAdam/hidden_6/bias/vAdam/output/kernel/vAdam/output/bias/v*=
Tin6
422*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_6584553??
?T
?
"__inference__wrapped_model_6583433	
inputF
4simple_model_hidden_1_matmul_readvariableop_resource:@C
5simple_model_hidden_1_biasadd_readvariableop_resource:@F
4simple_model_hidden_2_matmul_readvariableop_resource:@ C
5simple_model_hidden_2_biasadd_readvariableop_resource: F
4simple_model_hidden_3_matmul_readvariableop_resource: @C
5simple_model_hidden_3_biasadd_readvariableop_resource:@G
4simple_model_hidden_4_matmul_readvariableop_resource:	@?D
5simple_model_hidden_4_biasadd_readvariableop_resource:	?H
4simple_model_hidden_5_matmul_readvariableop_resource:
??D
5simple_model_hidden_5_biasadd_readvariableop_resource:	?G
4simple_model_hidden_6_matmul_readvariableop_resource:	?@C
5simple_model_hidden_6_biasadd_readvariableop_resource:@D
2simple_model_output_matmul_readvariableop_resource:@A
3simple_model_output_biasadd_readvariableop_resource:
identity??,simple_model/hidden_1/BiasAdd/ReadVariableOp?+simple_model/hidden_1/MatMul/ReadVariableOp?,simple_model/hidden_2/BiasAdd/ReadVariableOp?+simple_model/hidden_2/MatMul/ReadVariableOp?,simple_model/hidden_3/BiasAdd/ReadVariableOp?+simple_model/hidden_3/MatMul/ReadVariableOp?,simple_model/hidden_4/BiasAdd/ReadVariableOp?+simple_model/hidden_4/MatMul/ReadVariableOp?,simple_model/hidden_5/BiasAdd/ReadVariableOp?+simple_model/hidden_5/MatMul/ReadVariableOp?,simple_model/hidden_6/BiasAdd/ReadVariableOp?+simple_model/hidden_6/MatMul/ReadVariableOp?*simple_model/output/BiasAdd/ReadVariableOp?)simple_model/output/MatMul/ReadVariableOp?
+simple_model/hidden_1/MatMul/ReadVariableOpReadVariableOp4simple_model_hidden_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02-
+simple_model/hidden_1/MatMul/ReadVariableOp?
simple_model/hidden_1/MatMulMatMulinput3simple_model/hidden_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
simple_model/hidden_1/MatMul?
,simple_model/hidden_1/BiasAdd/ReadVariableOpReadVariableOp5simple_model_hidden_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,simple_model/hidden_1/BiasAdd/ReadVariableOp?
simple_model/hidden_1/BiasAddBiasAdd&simple_model/hidden_1/MatMul:product:04simple_model/hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
simple_model/hidden_1/BiasAdd?
simple_model/hidden_1/TanhTanh&simple_model/hidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
simple_model/hidden_1/Tanh?
+simple_model/hidden_2/MatMul/ReadVariableOpReadVariableOp4simple_model_hidden_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02-
+simple_model/hidden_2/MatMul/ReadVariableOp?
simple_model/hidden_2/MatMulMatMulsimple_model/hidden_1/Tanh:y:03simple_model/hidden_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
simple_model/hidden_2/MatMul?
,simple_model/hidden_2/BiasAdd/ReadVariableOpReadVariableOp5simple_model_hidden_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,simple_model/hidden_2/BiasAdd/ReadVariableOp?
simple_model/hidden_2/BiasAddBiasAdd&simple_model/hidden_2/MatMul:product:04simple_model/hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
simple_model/hidden_2/BiasAdd?
simple_model/hidden_2/TanhTanh&simple_model/hidden_2/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
simple_model/hidden_2/Tanh?
+simple_model/hidden_3/MatMul/ReadVariableOpReadVariableOp4simple_model_hidden_3_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02-
+simple_model/hidden_3/MatMul/ReadVariableOp?
simple_model/hidden_3/MatMulMatMulsimple_model/hidden_2/Tanh:y:03simple_model/hidden_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
simple_model/hidden_3/MatMul?
,simple_model/hidden_3/BiasAdd/ReadVariableOpReadVariableOp5simple_model_hidden_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,simple_model/hidden_3/BiasAdd/ReadVariableOp?
simple_model/hidden_3/BiasAddBiasAdd&simple_model/hidden_3/MatMul:product:04simple_model/hidden_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
simple_model/hidden_3/BiasAdd?
simple_model/hidden_3/TanhTanh&simple_model/hidden_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
simple_model/hidden_3/Tanh?
+simple_model/hidden_4/MatMul/ReadVariableOpReadVariableOp4simple_model_hidden_4_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02-
+simple_model/hidden_4/MatMul/ReadVariableOp?
simple_model/hidden_4/MatMulMatMulsimple_model/hidden_3/Tanh:y:03simple_model/hidden_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_model/hidden_4/MatMul?
,simple_model/hidden_4/BiasAdd/ReadVariableOpReadVariableOp5simple_model_hidden_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,simple_model/hidden_4/BiasAdd/ReadVariableOp?
simple_model/hidden_4/BiasAddBiasAdd&simple_model/hidden_4/MatMul:product:04simple_model/hidden_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_model/hidden_4/BiasAdd?
simple_model/hidden_4/TanhTanh&simple_model/hidden_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
simple_model/hidden_4/Tanh?
+simple_model/hidden_5/MatMul/ReadVariableOpReadVariableOp4simple_model_hidden_5_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+simple_model/hidden_5/MatMul/ReadVariableOp?
simple_model/hidden_5/MatMulMatMulsimple_model/hidden_4/Tanh:y:03simple_model/hidden_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_model/hidden_5/MatMul?
,simple_model/hidden_5/BiasAdd/ReadVariableOpReadVariableOp5simple_model_hidden_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,simple_model/hidden_5/BiasAdd/ReadVariableOp?
simple_model/hidden_5/BiasAddBiasAdd&simple_model/hidden_5/MatMul:product:04simple_model/hidden_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_model/hidden_5/BiasAdd?
simple_model/hidden_5/TanhTanh&simple_model/hidden_5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
simple_model/hidden_5/Tanh?
+simple_model/hidden_6/MatMul/ReadVariableOpReadVariableOp4simple_model_hidden_6_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02-
+simple_model/hidden_6/MatMul/ReadVariableOp?
simple_model/hidden_6/MatMulMatMulsimple_model/hidden_5/Tanh:y:03simple_model/hidden_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
simple_model/hidden_6/MatMul?
,simple_model/hidden_6/BiasAdd/ReadVariableOpReadVariableOp5simple_model_hidden_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,simple_model/hidden_6/BiasAdd/ReadVariableOp?
simple_model/hidden_6/BiasAddBiasAdd&simple_model/hidden_6/MatMul:product:04simple_model/hidden_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
simple_model/hidden_6/BiasAdd?
simple_model/hidden_6/TanhTanh&simple_model/hidden_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
simple_model/hidden_6/Tanh?
)simple_model/output/MatMul/ReadVariableOpReadVariableOp2simple_model_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02+
)simple_model/output/MatMul/ReadVariableOp?
simple_model/output/MatMulMatMulsimple_model/hidden_6/Tanh:y:01simple_model/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
simple_model/output/MatMul?
*simple_model/output/BiasAdd/ReadVariableOpReadVariableOp3simple_model_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*simple_model/output/BiasAdd/ReadVariableOp?
simple_model/output/BiasAddBiasAdd$simple_model/output/MatMul:product:02simple_model/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
simple_model/output/BiasAdd?
IdentityIdentity$simple_model/output/BiasAdd:output:0-^simple_model/hidden_1/BiasAdd/ReadVariableOp,^simple_model/hidden_1/MatMul/ReadVariableOp-^simple_model/hidden_2/BiasAdd/ReadVariableOp,^simple_model/hidden_2/MatMul/ReadVariableOp-^simple_model/hidden_3/BiasAdd/ReadVariableOp,^simple_model/hidden_3/MatMul/ReadVariableOp-^simple_model/hidden_4/BiasAdd/ReadVariableOp,^simple_model/hidden_4/MatMul/ReadVariableOp-^simple_model/hidden_5/BiasAdd/ReadVariableOp,^simple_model/hidden_5/MatMul/ReadVariableOp-^simple_model/hidden_6/BiasAdd/ReadVariableOp,^simple_model/hidden_6/MatMul/ReadVariableOp+^simple_model/output/BiasAdd/ReadVariableOp*^simple_model/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 2\
,simple_model/hidden_1/BiasAdd/ReadVariableOp,simple_model/hidden_1/BiasAdd/ReadVariableOp2Z
+simple_model/hidden_1/MatMul/ReadVariableOp+simple_model/hidden_1/MatMul/ReadVariableOp2\
,simple_model/hidden_2/BiasAdd/ReadVariableOp,simple_model/hidden_2/BiasAdd/ReadVariableOp2Z
+simple_model/hidden_2/MatMul/ReadVariableOp+simple_model/hidden_2/MatMul/ReadVariableOp2\
,simple_model/hidden_3/BiasAdd/ReadVariableOp,simple_model/hidden_3/BiasAdd/ReadVariableOp2Z
+simple_model/hidden_3/MatMul/ReadVariableOp+simple_model/hidden_3/MatMul/ReadVariableOp2\
,simple_model/hidden_4/BiasAdd/ReadVariableOp,simple_model/hidden_4/BiasAdd/ReadVariableOp2Z
+simple_model/hidden_4/MatMul/ReadVariableOp+simple_model/hidden_4/MatMul/ReadVariableOp2\
,simple_model/hidden_5/BiasAdd/ReadVariableOp,simple_model/hidden_5/BiasAdd/ReadVariableOp2Z
+simple_model/hidden_5/MatMul/ReadVariableOp+simple_model/hidden_5/MatMul/ReadVariableOp2\
,simple_model/hidden_6/BiasAdd/ReadVariableOp,simple_model/hidden_6/BiasAdd/ReadVariableOp2Z
+simple_model/hidden_6/MatMul/ReadVariableOp+simple_model/hidden_6/MatMul/ReadVariableOp2X
*simple_model/output/BiasAdd/ReadVariableOp*simple_model/output/BiasAdd/ReadVariableOp2V
)simple_model/output/MatMul/ReadVariableOp)simple_model/output/MatMul/ReadVariableOp:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?	
?
C__inference_output_layer_call_and_return_conditional_losses_6583552

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
%__inference_signature_wrapper_6583917	
input
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: @
	unknown_4:@
	unknown_5:	@?
	unknown_6:	?
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?@

unknown_10:@

unknown_11:@

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_65834332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?A
?

I__inference_simple_model_layer_call_and_return_conditional_losses_6584087

inputs9
'hidden_1_matmul_readvariableop_resource:@6
(hidden_1_biasadd_readvariableop_resource:@9
'hidden_2_matmul_readvariableop_resource:@ 6
(hidden_2_biasadd_readvariableop_resource: 9
'hidden_3_matmul_readvariableop_resource: @6
(hidden_3_biasadd_readvariableop_resource:@:
'hidden_4_matmul_readvariableop_resource:	@?7
(hidden_4_biasadd_readvariableop_resource:	?;
'hidden_5_matmul_readvariableop_resource:
??7
(hidden_5_biasadd_readvariableop_resource:	?:
'hidden_6_matmul_readvariableop_resource:	?@6
(hidden_6_biasadd_readvariableop_resource:@7
%output_matmul_readvariableop_resource:@4
&output_biasadd_readvariableop_resource:
identity??hidden_1/BiasAdd/ReadVariableOp?hidden_1/MatMul/ReadVariableOp?hidden_2/BiasAdd/ReadVariableOp?hidden_2/MatMul/ReadVariableOp?hidden_3/BiasAdd/ReadVariableOp?hidden_3/MatMul/ReadVariableOp?hidden_4/BiasAdd/ReadVariableOp?hidden_4/MatMul/ReadVariableOp?hidden_5/BiasAdd/ReadVariableOp?hidden_5/MatMul/ReadVariableOp?hidden_6/BiasAdd/ReadVariableOp?hidden_6/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
hidden_1/MatMul/ReadVariableOpReadVariableOp'hidden_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
hidden_1/MatMul/ReadVariableOp?
hidden_1/MatMulMatMulinputs&hidden_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
hidden_1/MatMul?
hidden_1/BiasAdd/ReadVariableOpReadVariableOp(hidden_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
hidden_1/BiasAdd/ReadVariableOp?
hidden_1/BiasAddBiasAddhidden_1/MatMul:product:0'hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
hidden_1/BiasAdds
hidden_1/TanhTanhhidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
hidden_1/Tanh?
hidden_2/MatMul/ReadVariableOpReadVariableOp'hidden_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02 
hidden_2/MatMul/ReadVariableOp?
hidden_2/MatMulMatMulhidden_1/Tanh:y:0&hidden_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
hidden_2/MatMul?
hidden_2/BiasAdd/ReadVariableOpReadVariableOp(hidden_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
hidden_2/BiasAdd/ReadVariableOp?
hidden_2/BiasAddBiasAddhidden_2/MatMul:product:0'hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
hidden_2/BiasAdds
hidden_2/TanhTanhhidden_2/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
hidden_2/Tanh?
hidden_3/MatMul/ReadVariableOpReadVariableOp'hidden_3_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02 
hidden_3/MatMul/ReadVariableOp?
hidden_3/MatMulMatMulhidden_2/Tanh:y:0&hidden_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
hidden_3/MatMul?
hidden_3/BiasAdd/ReadVariableOpReadVariableOp(hidden_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
hidden_3/BiasAdd/ReadVariableOp?
hidden_3/BiasAddBiasAddhidden_3/MatMul:product:0'hidden_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
hidden_3/BiasAdds
hidden_3/TanhTanhhidden_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
hidden_3/Tanh?
hidden_4/MatMul/ReadVariableOpReadVariableOp'hidden_4_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02 
hidden_4/MatMul/ReadVariableOp?
hidden_4/MatMulMatMulhidden_3/Tanh:y:0&hidden_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hidden_4/MatMul?
hidden_4/BiasAdd/ReadVariableOpReadVariableOp(hidden_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
hidden_4/BiasAdd/ReadVariableOp?
hidden_4/BiasAddBiasAddhidden_4/MatMul:product:0'hidden_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hidden_4/BiasAddt
hidden_4/TanhTanhhidden_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
hidden_4/Tanh?
hidden_5/MatMul/ReadVariableOpReadVariableOp'hidden_5_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
hidden_5/MatMul/ReadVariableOp?
hidden_5/MatMulMatMulhidden_4/Tanh:y:0&hidden_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hidden_5/MatMul?
hidden_5/BiasAdd/ReadVariableOpReadVariableOp(hidden_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
hidden_5/BiasAdd/ReadVariableOp?
hidden_5/BiasAddBiasAddhidden_5/MatMul:product:0'hidden_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hidden_5/BiasAddt
hidden_5/TanhTanhhidden_5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
hidden_5/Tanh?
hidden_6/MatMul/ReadVariableOpReadVariableOp'hidden_6_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02 
hidden_6/MatMul/ReadVariableOp?
hidden_6/MatMulMatMulhidden_5/Tanh:y:0&hidden_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
hidden_6/MatMul?
hidden_6/BiasAdd/ReadVariableOpReadVariableOp(hidden_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
hidden_6/BiasAdd/ReadVariableOp?
hidden_6/BiasAddBiasAddhidden_6/MatMul:product:0'hidden_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
hidden_6/BiasAdds
hidden_6/TanhTanhhidden_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
hidden_6/Tanh?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMulhidden_6/Tanh:y:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/BiasAdd?
IdentityIdentityoutput/BiasAdd:output:0 ^hidden_1/BiasAdd/ReadVariableOp^hidden_1/MatMul/ReadVariableOp ^hidden_2/BiasAdd/ReadVariableOp^hidden_2/MatMul/ReadVariableOp ^hidden_3/BiasAdd/ReadVariableOp^hidden_3/MatMul/ReadVariableOp ^hidden_4/BiasAdd/ReadVariableOp^hidden_4/MatMul/ReadVariableOp ^hidden_5/BiasAdd/ReadVariableOp^hidden_5/MatMul/ReadVariableOp ^hidden_6/BiasAdd/ReadVariableOp^hidden_6/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 2B
hidden_1/BiasAdd/ReadVariableOphidden_1/BiasAdd/ReadVariableOp2@
hidden_1/MatMul/ReadVariableOphidden_1/MatMul/ReadVariableOp2B
hidden_2/BiasAdd/ReadVariableOphidden_2/BiasAdd/ReadVariableOp2@
hidden_2/MatMul/ReadVariableOphidden_2/MatMul/ReadVariableOp2B
hidden_3/BiasAdd/ReadVariableOphidden_3/BiasAdd/ReadVariableOp2@
hidden_3/MatMul/ReadVariableOphidden_3/MatMul/ReadVariableOp2B
hidden_4/BiasAdd/ReadVariableOphidden_4/BiasAdd/ReadVariableOp2@
hidden_4/MatMul/ReadVariableOphidden_4/MatMul/ReadVariableOp2B
hidden_5/BiasAdd/ReadVariableOphidden_5/BiasAdd/ReadVariableOp2@
hidden_5/MatMul/ReadVariableOphidden_5/MatMul/ReadVariableOp2B
hidden_6/BiasAdd/ReadVariableOphidden_6/BiasAdd/ReadVariableOp2@
hidden_6/MatMul/ReadVariableOphidden_6/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_simple_model_layer_call_fn_6583798	
input
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: @
	unknown_4:@
	unknown_5:	@?
	unknown_6:	?
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?@

unknown_10:@

unknown_11:@

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_simple_model_layer_call_and_return_conditional_losses_65837342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?

?
E__inference_hidden_1_layer_call_and_return_conditional_losses_6584107

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_output_layer_call_fn_6584216

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_65835522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
.__inference_simple_model_layer_call_fn_6583983

inputs
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: @
	unknown_4:@
	unknown_5:	@?
	unknown_6:	?
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?@

unknown_10:@

unknown_11:@

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_simple_model_layer_call_and_return_conditional_losses_65837342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_hidden_5_layer_call_and_return_conditional_losses_6584187

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_hidden_5_layer_call_and_return_conditional_losses_6583519

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?A
?

I__inference_simple_model_layer_call_and_return_conditional_losses_6584035

inputs9
'hidden_1_matmul_readvariableop_resource:@6
(hidden_1_biasadd_readvariableop_resource:@9
'hidden_2_matmul_readvariableop_resource:@ 6
(hidden_2_biasadd_readvariableop_resource: 9
'hidden_3_matmul_readvariableop_resource: @6
(hidden_3_biasadd_readvariableop_resource:@:
'hidden_4_matmul_readvariableop_resource:	@?7
(hidden_4_biasadd_readvariableop_resource:	?;
'hidden_5_matmul_readvariableop_resource:
??7
(hidden_5_biasadd_readvariableop_resource:	?:
'hidden_6_matmul_readvariableop_resource:	?@6
(hidden_6_biasadd_readvariableop_resource:@7
%output_matmul_readvariableop_resource:@4
&output_biasadd_readvariableop_resource:
identity??hidden_1/BiasAdd/ReadVariableOp?hidden_1/MatMul/ReadVariableOp?hidden_2/BiasAdd/ReadVariableOp?hidden_2/MatMul/ReadVariableOp?hidden_3/BiasAdd/ReadVariableOp?hidden_3/MatMul/ReadVariableOp?hidden_4/BiasAdd/ReadVariableOp?hidden_4/MatMul/ReadVariableOp?hidden_5/BiasAdd/ReadVariableOp?hidden_5/MatMul/ReadVariableOp?hidden_6/BiasAdd/ReadVariableOp?hidden_6/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
hidden_1/MatMul/ReadVariableOpReadVariableOp'hidden_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
hidden_1/MatMul/ReadVariableOp?
hidden_1/MatMulMatMulinputs&hidden_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
hidden_1/MatMul?
hidden_1/BiasAdd/ReadVariableOpReadVariableOp(hidden_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
hidden_1/BiasAdd/ReadVariableOp?
hidden_1/BiasAddBiasAddhidden_1/MatMul:product:0'hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
hidden_1/BiasAdds
hidden_1/TanhTanhhidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
hidden_1/Tanh?
hidden_2/MatMul/ReadVariableOpReadVariableOp'hidden_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02 
hidden_2/MatMul/ReadVariableOp?
hidden_2/MatMulMatMulhidden_1/Tanh:y:0&hidden_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
hidden_2/MatMul?
hidden_2/BiasAdd/ReadVariableOpReadVariableOp(hidden_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
hidden_2/BiasAdd/ReadVariableOp?
hidden_2/BiasAddBiasAddhidden_2/MatMul:product:0'hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
hidden_2/BiasAdds
hidden_2/TanhTanhhidden_2/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
hidden_2/Tanh?
hidden_3/MatMul/ReadVariableOpReadVariableOp'hidden_3_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02 
hidden_3/MatMul/ReadVariableOp?
hidden_3/MatMulMatMulhidden_2/Tanh:y:0&hidden_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
hidden_3/MatMul?
hidden_3/BiasAdd/ReadVariableOpReadVariableOp(hidden_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
hidden_3/BiasAdd/ReadVariableOp?
hidden_3/BiasAddBiasAddhidden_3/MatMul:product:0'hidden_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
hidden_3/BiasAdds
hidden_3/TanhTanhhidden_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
hidden_3/Tanh?
hidden_4/MatMul/ReadVariableOpReadVariableOp'hidden_4_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02 
hidden_4/MatMul/ReadVariableOp?
hidden_4/MatMulMatMulhidden_3/Tanh:y:0&hidden_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hidden_4/MatMul?
hidden_4/BiasAdd/ReadVariableOpReadVariableOp(hidden_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
hidden_4/BiasAdd/ReadVariableOp?
hidden_4/BiasAddBiasAddhidden_4/MatMul:product:0'hidden_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hidden_4/BiasAddt
hidden_4/TanhTanhhidden_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
hidden_4/Tanh?
hidden_5/MatMul/ReadVariableOpReadVariableOp'hidden_5_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
hidden_5/MatMul/ReadVariableOp?
hidden_5/MatMulMatMulhidden_4/Tanh:y:0&hidden_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hidden_5/MatMul?
hidden_5/BiasAdd/ReadVariableOpReadVariableOp(hidden_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
hidden_5/BiasAdd/ReadVariableOp?
hidden_5/BiasAddBiasAddhidden_5/MatMul:product:0'hidden_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hidden_5/BiasAddt
hidden_5/TanhTanhhidden_5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
hidden_5/Tanh?
hidden_6/MatMul/ReadVariableOpReadVariableOp'hidden_6_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02 
hidden_6/MatMul/ReadVariableOp?
hidden_6/MatMulMatMulhidden_5/Tanh:y:0&hidden_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
hidden_6/MatMul?
hidden_6/BiasAdd/ReadVariableOpReadVariableOp(hidden_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
hidden_6/BiasAdd/ReadVariableOp?
hidden_6/BiasAddBiasAddhidden_6/MatMul:product:0'hidden_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
hidden_6/BiasAdds
hidden_6/TanhTanhhidden_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
hidden_6/Tanh?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMulhidden_6/Tanh:y:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/BiasAdd?
IdentityIdentityoutput/BiasAdd:output:0 ^hidden_1/BiasAdd/ReadVariableOp^hidden_1/MatMul/ReadVariableOp ^hidden_2/BiasAdd/ReadVariableOp^hidden_2/MatMul/ReadVariableOp ^hidden_3/BiasAdd/ReadVariableOp^hidden_3/MatMul/ReadVariableOp ^hidden_4/BiasAdd/ReadVariableOp^hidden_4/MatMul/ReadVariableOp ^hidden_5/BiasAdd/ReadVariableOp^hidden_5/MatMul/ReadVariableOp ^hidden_6/BiasAdd/ReadVariableOp^hidden_6/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 2B
hidden_1/BiasAdd/ReadVariableOphidden_1/BiasAdd/ReadVariableOp2@
hidden_1/MatMul/ReadVariableOphidden_1/MatMul/ReadVariableOp2B
hidden_2/BiasAdd/ReadVariableOphidden_2/BiasAdd/ReadVariableOp2@
hidden_2/MatMul/ReadVariableOphidden_2/MatMul/ReadVariableOp2B
hidden_3/BiasAdd/ReadVariableOphidden_3/BiasAdd/ReadVariableOp2@
hidden_3/MatMul/ReadVariableOphidden_3/MatMul/ReadVariableOp2B
hidden_4/BiasAdd/ReadVariableOphidden_4/BiasAdd/ReadVariableOp2@
hidden_4/MatMul/ReadVariableOphidden_4/MatMul/ReadVariableOp2B
hidden_5/BiasAdd/ReadVariableOphidden_5/BiasAdd/ReadVariableOp2@
hidden_5/MatMul/ReadVariableOphidden_5/MatMul/ReadVariableOp2B
hidden_6/BiasAdd/ReadVariableOphidden_6/BiasAdd/ReadVariableOp2@
hidden_6/MatMul/ReadVariableOphidden_6/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_hidden_2_layer_call_and_return_conditional_losses_6584127

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
*__inference_hidden_4_layer_call_fn_6584156

inputs
unknown:	@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_4_layer_call_and_return_conditional_losses_65835022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
E__inference_hidden_4_layer_call_and_return_conditional_losses_6584167

inputs1
matmul_readvariableop_resource:	@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
E__inference_hidden_4_layer_call_and_return_conditional_losses_6583502

inputs1
matmul_readvariableop_resource:	@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
#__inference__traced_restore_6584553
file_prefix2
 assignvariableop_hidden_1_kernel:@.
 assignvariableop_1_hidden_1_bias:@4
"assignvariableop_2_hidden_2_kernel:@ .
 assignvariableop_3_hidden_2_bias: 4
"assignvariableop_4_hidden_3_kernel: @.
 assignvariableop_5_hidden_3_bias:@5
"assignvariableop_6_hidden_4_kernel:	@?/
 assignvariableop_7_hidden_4_bias:	?6
"assignvariableop_8_hidden_5_kernel:
??/
 assignvariableop_9_hidden_5_bias:	?6
#assignvariableop_10_hidden_6_kernel:	?@/
!assignvariableop_11_hidden_6_bias:@3
!assignvariableop_12_output_kernel:@-
assignvariableop_13_output_bias:'
assignvariableop_14_adam_iter:	 )
assignvariableop_15_adam_beta_1: )
assignvariableop_16_adam_beta_2: (
assignvariableop_17_adam_decay: 0
&assignvariableop_18_adam_learning_rate: #
assignvariableop_19_total: #
assignvariableop_20_count: <
*assignvariableop_21_adam_hidden_1_kernel_m:@6
(assignvariableop_22_adam_hidden_1_bias_m:@<
*assignvariableop_23_adam_hidden_2_kernel_m:@ 6
(assignvariableop_24_adam_hidden_2_bias_m: <
*assignvariableop_25_adam_hidden_3_kernel_m: @6
(assignvariableop_26_adam_hidden_3_bias_m:@=
*assignvariableop_27_adam_hidden_4_kernel_m:	@?7
(assignvariableop_28_adam_hidden_4_bias_m:	?>
*assignvariableop_29_adam_hidden_5_kernel_m:
??7
(assignvariableop_30_adam_hidden_5_bias_m:	?=
*assignvariableop_31_adam_hidden_6_kernel_m:	?@6
(assignvariableop_32_adam_hidden_6_bias_m:@:
(assignvariableop_33_adam_output_kernel_m:@4
&assignvariableop_34_adam_output_bias_m:<
*assignvariableop_35_adam_hidden_1_kernel_v:@6
(assignvariableop_36_adam_hidden_1_bias_v:@<
*assignvariableop_37_adam_hidden_2_kernel_v:@ 6
(assignvariableop_38_adam_hidden_2_bias_v: <
*assignvariableop_39_adam_hidden_3_kernel_v: @6
(assignvariableop_40_adam_hidden_3_bias_v:@=
*assignvariableop_41_adam_hidden_4_kernel_v:	@?7
(assignvariableop_42_adam_hidden_4_bias_v:	?>
*assignvariableop_43_adam_hidden_5_kernel_v:
??7
(assignvariableop_44_adam_hidden_5_bias_v:	?=
*assignvariableop_45_adam_hidden_6_kernel_v:	?@6
(assignvariableop_46_adam_hidden_6_bias_v:@:
(assignvariableop_47_adam_output_kernel_v:@4
&assignvariableop_48_adam_output_bias_v:
identity_50??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*?
value?B?2B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_hidden_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_hidden_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_hidden_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_hidden_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_hidden_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_hidden_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_hidden_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_hidden_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_hidden_5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_hidden_5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_hidden_6_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_hidden_6_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_output_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_output_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_hidden_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_hidden_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_hidden_2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_hidden_2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_hidden_3_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_hidden_3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_hidden_4_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_hidden_4_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_hidden_5_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_hidden_5_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_hidden_6_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_hidden_6_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_output_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_output_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_hidden_1_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_hidden_1_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_hidden_2_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_hidden_2_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_hidden_3_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_hidden_3_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_hidden_4_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_hidden_4_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_hidden_5_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_hidden_5_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_hidden_6_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_hidden_6_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_output_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp&assignvariableop_48_adam_output_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_489
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?	
Identity_49Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_49?	
Identity_50IdentityIdentity_49:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_50"#
identity_50Identity_50:output:0*w
_input_shapesf
d: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

?
E__inference_hidden_3_layer_call_and_return_conditional_losses_6584147

inputs0
matmul_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
.__inference_simple_model_layer_call_fn_6583950

inputs
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: @
	unknown_4:@
	unknown_5:	@?
	unknown_6:	?
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?@

unknown_10:@

unknown_11:@

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_simple_model_layer_call_and_return_conditional_losses_65835592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_hidden_6_layer_call_and_return_conditional_losses_6583536

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?&
?
I__inference_simple_model_layer_call_and_return_conditional_losses_6583734

inputs"
hidden_1_6583698:@
hidden_1_6583700:@"
hidden_2_6583703:@ 
hidden_2_6583705: "
hidden_3_6583708: @
hidden_3_6583710:@#
hidden_4_6583713:	@?
hidden_4_6583715:	?$
hidden_5_6583718:
??
hidden_5_6583720:	?#
hidden_6_6583723:	?@
hidden_6_6583725:@ 
output_6583728:@
output_6583730:
identity?? hidden_1/StatefulPartitionedCall? hidden_2/StatefulPartitionedCall? hidden_3/StatefulPartitionedCall? hidden_4/StatefulPartitionedCall? hidden_5/StatefulPartitionedCall? hidden_6/StatefulPartitionedCall?output/StatefulPartitionedCall?
 hidden_1/StatefulPartitionedCallStatefulPartitionedCallinputshidden_1_6583698hidden_1_6583700*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_1_layer_call_and_return_conditional_losses_65834512"
 hidden_1/StatefulPartitionedCall?
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0hidden_2_6583703hidden_2_6583705*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_2_layer_call_and_return_conditional_losses_65834682"
 hidden_2/StatefulPartitionedCall?
 hidden_3/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0hidden_3_6583708hidden_3_6583710*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_3_layer_call_and_return_conditional_losses_65834852"
 hidden_3/StatefulPartitionedCall?
 hidden_4/StatefulPartitionedCallStatefulPartitionedCall)hidden_3/StatefulPartitionedCall:output:0hidden_4_6583713hidden_4_6583715*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_4_layer_call_and_return_conditional_losses_65835022"
 hidden_4/StatefulPartitionedCall?
 hidden_5/StatefulPartitionedCallStatefulPartitionedCall)hidden_4/StatefulPartitionedCall:output:0hidden_5_6583718hidden_5_6583720*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_5_layer_call_and_return_conditional_losses_65835192"
 hidden_5/StatefulPartitionedCall?
 hidden_6/StatefulPartitionedCallStatefulPartitionedCall)hidden_5/StatefulPartitionedCall:output:0hidden_6_6583723hidden_6_6583725*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_6_layer_call_and_return_conditional_losses_65835362"
 hidden_6/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall)hidden_6/StatefulPartitionedCall:output:0output_6583728output_6583730*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_65835522 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0!^hidden_1/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall!^hidden_3/StatefulPartitionedCall!^hidden_4/StatefulPartitionedCall!^hidden_5/StatefulPartitionedCall!^hidden_6/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2D
 hidden_3/StatefulPartitionedCall hidden_3/StatefulPartitionedCall2D
 hidden_4/StatefulPartitionedCall hidden_4/StatefulPartitionedCall2D
 hidden_5/StatefulPartitionedCall hidden_5/StatefulPartitionedCall2D
 hidden_6/StatefulPartitionedCall hidden_6/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_hidden_6_layer_call_fn_6584196

inputs
unknown:	?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_6_layer_call_and_return_conditional_losses_65835362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_hidden_5_layer_call_fn_6584176

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_5_layer_call_and_return_conditional_losses_65835192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_hidden_2_layer_call_fn_6584116

inputs
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_2_layer_call_and_return_conditional_losses_65834682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?&
?
I__inference_simple_model_layer_call_and_return_conditional_losses_6583876	
input"
hidden_1_6583840:@
hidden_1_6583842:@"
hidden_2_6583845:@ 
hidden_2_6583847: "
hidden_3_6583850: @
hidden_3_6583852:@#
hidden_4_6583855:	@?
hidden_4_6583857:	?$
hidden_5_6583860:
??
hidden_5_6583862:	?#
hidden_6_6583865:	?@
hidden_6_6583867:@ 
output_6583870:@
output_6583872:
identity?? hidden_1/StatefulPartitionedCall? hidden_2/StatefulPartitionedCall? hidden_3/StatefulPartitionedCall? hidden_4/StatefulPartitionedCall? hidden_5/StatefulPartitionedCall? hidden_6/StatefulPartitionedCall?output/StatefulPartitionedCall?
 hidden_1/StatefulPartitionedCallStatefulPartitionedCallinputhidden_1_6583840hidden_1_6583842*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_1_layer_call_and_return_conditional_losses_65834512"
 hidden_1/StatefulPartitionedCall?
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0hidden_2_6583845hidden_2_6583847*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_2_layer_call_and_return_conditional_losses_65834682"
 hidden_2/StatefulPartitionedCall?
 hidden_3/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0hidden_3_6583850hidden_3_6583852*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_3_layer_call_and_return_conditional_losses_65834852"
 hidden_3/StatefulPartitionedCall?
 hidden_4/StatefulPartitionedCallStatefulPartitionedCall)hidden_3/StatefulPartitionedCall:output:0hidden_4_6583855hidden_4_6583857*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_4_layer_call_and_return_conditional_losses_65835022"
 hidden_4/StatefulPartitionedCall?
 hidden_5/StatefulPartitionedCallStatefulPartitionedCall)hidden_4/StatefulPartitionedCall:output:0hidden_5_6583860hidden_5_6583862*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_5_layer_call_and_return_conditional_losses_65835192"
 hidden_5/StatefulPartitionedCall?
 hidden_6/StatefulPartitionedCallStatefulPartitionedCall)hidden_5/StatefulPartitionedCall:output:0hidden_6_6583865hidden_6_6583867*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_6_layer_call_and_return_conditional_losses_65835362"
 hidden_6/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall)hidden_6/StatefulPartitionedCall:output:0output_6583870output_6583872*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_65835522 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0!^hidden_1/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall!^hidden_3/StatefulPartitionedCall!^hidden_4/StatefulPartitionedCall!^hidden_5/StatefulPartitionedCall!^hidden_6/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2D
 hidden_3/StatefulPartitionedCall hidden_3/StatefulPartitionedCall2D
 hidden_4/StatefulPartitionedCall hidden_4/StatefulPartitionedCall2D
 hidden_5/StatefulPartitionedCall hidden_5/StatefulPartitionedCall2D
 hidden_6/StatefulPartitionedCall hidden_6/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?	
?
C__inference_output_layer_call_and_return_conditional_losses_6584226

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?d
?
 __inference__traced_save_6584396
file_prefix.
*savev2_hidden_1_kernel_read_readvariableop,
(savev2_hidden_1_bias_read_readvariableop.
*savev2_hidden_2_kernel_read_readvariableop,
(savev2_hidden_2_bias_read_readvariableop.
*savev2_hidden_3_kernel_read_readvariableop,
(savev2_hidden_3_bias_read_readvariableop.
*savev2_hidden_4_kernel_read_readvariableop,
(savev2_hidden_4_bias_read_readvariableop.
*savev2_hidden_5_kernel_read_readvariableop,
(savev2_hidden_5_bias_read_readvariableop.
*savev2_hidden_6_kernel_read_readvariableop,
(savev2_hidden_6_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_hidden_1_kernel_m_read_readvariableop3
/savev2_adam_hidden_1_bias_m_read_readvariableop5
1savev2_adam_hidden_2_kernel_m_read_readvariableop3
/savev2_adam_hidden_2_bias_m_read_readvariableop5
1savev2_adam_hidden_3_kernel_m_read_readvariableop3
/savev2_adam_hidden_3_bias_m_read_readvariableop5
1savev2_adam_hidden_4_kernel_m_read_readvariableop3
/savev2_adam_hidden_4_bias_m_read_readvariableop5
1savev2_adam_hidden_5_kernel_m_read_readvariableop3
/savev2_adam_hidden_5_bias_m_read_readvariableop5
1savev2_adam_hidden_6_kernel_m_read_readvariableop3
/savev2_adam_hidden_6_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop5
1savev2_adam_hidden_1_kernel_v_read_readvariableop3
/savev2_adam_hidden_1_bias_v_read_readvariableop5
1savev2_adam_hidden_2_kernel_v_read_readvariableop3
/savev2_adam_hidden_2_bias_v_read_readvariableop5
1savev2_adam_hidden_3_kernel_v_read_readvariableop3
/savev2_adam_hidden_3_bias_v_read_readvariableop5
1savev2_adam_hidden_4_kernel_v_read_readvariableop3
/savev2_adam_hidden_4_bias_v_read_readvariableop5
1savev2_adam_hidden_5_kernel_v_read_readvariableop3
/savev2_adam_hidden_5_bias_v_read_readvariableop5
1savev2_adam_hidden_6_kernel_v_read_readvariableop3
/savev2_adam_hidden_6_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*?
value?B?2B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_hidden_1_kernel_read_readvariableop(savev2_hidden_1_bias_read_readvariableop*savev2_hidden_2_kernel_read_readvariableop(savev2_hidden_2_bias_read_readvariableop*savev2_hidden_3_kernel_read_readvariableop(savev2_hidden_3_bias_read_readvariableop*savev2_hidden_4_kernel_read_readvariableop(savev2_hidden_4_bias_read_readvariableop*savev2_hidden_5_kernel_read_readvariableop(savev2_hidden_5_bias_read_readvariableop*savev2_hidden_6_kernel_read_readvariableop(savev2_hidden_6_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_hidden_1_kernel_m_read_readvariableop/savev2_adam_hidden_1_bias_m_read_readvariableop1savev2_adam_hidden_2_kernel_m_read_readvariableop/savev2_adam_hidden_2_bias_m_read_readvariableop1savev2_adam_hidden_3_kernel_m_read_readvariableop/savev2_adam_hidden_3_bias_m_read_readvariableop1savev2_adam_hidden_4_kernel_m_read_readvariableop/savev2_adam_hidden_4_bias_m_read_readvariableop1savev2_adam_hidden_5_kernel_m_read_readvariableop/savev2_adam_hidden_5_bias_m_read_readvariableop1savev2_adam_hidden_6_kernel_m_read_readvariableop/savev2_adam_hidden_6_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop1savev2_adam_hidden_1_kernel_v_read_readvariableop/savev2_adam_hidden_1_bias_v_read_readvariableop1savev2_adam_hidden_2_kernel_v_read_readvariableop/savev2_adam_hidden_2_bias_v_read_readvariableop1savev2_adam_hidden_3_kernel_v_read_readvariableop/savev2_adam_hidden_3_bias_v_read_readvariableop1savev2_adam_hidden_4_kernel_v_read_readvariableop/savev2_adam_hidden_4_bias_v_read_readvariableop1savev2_adam_hidden_5_kernel_v_read_readvariableop/savev2_adam_hidden_5_bias_v_read_readvariableop1savev2_adam_hidden_6_kernel_v_read_readvariableop/savev2_adam_hidden_6_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *@
dtypes6
422	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@ : : @:@:	@?:?:
??:?:	?@:@:@:: : : : : : : :@:@:@ : : @:@:	@?:?:
??:?:	?@:@:@::@:@:@ : : @:@:	@?:?:
??:?:	?@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: @: 

_output_shapes
:@:%!

_output_shapes
:	@?:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: @: 

_output_shapes
:@:%!

_output_shapes
:	@?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:% !

_output_shapes
:	?@: !

_output_shapes
:@:$" 

_output_shapes

:@: #

_output_shapes
::$$ 

_output_shapes

:@: %

_output_shapes
:@:$& 

_output_shapes

:@ : '

_output_shapes
: :$( 

_output_shapes

: @: )

_output_shapes
:@:%*!

_output_shapes
:	@?:!+

_output_shapes	
:?:&,"
 
_output_shapes
:
??:!-

_output_shapes	
:?:%.!

_output_shapes
:	?@: /

_output_shapes
:@:$0 

_output_shapes

:@: 1

_output_shapes
::2

_output_shapes
: 
?

?
E__inference_hidden_3_layer_call_and_return_conditional_losses_6583485

inputs0
matmul_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
E__inference_hidden_2_layer_call_and_return_conditional_losses_6583468

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
E__inference_hidden_6_layer_call_and_return_conditional_losses_6584207

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_hidden_1_layer_call_and_return_conditional_losses_6583451

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_simple_model_layer_call_fn_6583590	
input
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: @
	unknown_4:@
	unknown_5:	@?
	unknown_6:	?
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?@

unknown_10:@

unknown_11:@

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_simple_model_layer_call_and_return_conditional_losses_65835592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?
?
*__inference_hidden_3_layer_call_fn_6584136

inputs
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_3_layer_call_and_return_conditional_losses_65834852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?&
?
I__inference_simple_model_layer_call_and_return_conditional_losses_6583559

inputs"
hidden_1_6583452:@
hidden_1_6583454:@"
hidden_2_6583469:@ 
hidden_2_6583471: "
hidden_3_6583486: @
hidden_3_6583488:@#
hidden_4_6583503:	@?
hidden_4_6583505:	?$
hidden_5_6583520:
??
hidden_5_6583522:	?#
hidden_6_6583537:	?@
hidden_6_6583539:@ 
output_6583553:@
output_6583555:
identity?? hidden_1/StatefulPartitionedCall? hidden_2/StatefulPartitionedCall? hidden_3/StatefulPartitionedCall? hidden_4/StatefulPartitionedCall? hidden_5/StatefulPartitionedCall? hidden_6/StatefulPartitionedCall?output/StatefulPartitionedCall?
 hidden_1/StatefulPartitionedCallStatefulPartitionedCallinputshidden_1_6583452hidden_1_6583454*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_1_layer_call_and_return_conditional_losses_65834512"
 hidden_1/StatefulPartitionedCall?
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0hidden_2_6583469hidden_2_6583471*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_2_layer_call_and_return_conditional_losses_65834682"
 hidden_2/StatefulPartitionedCall?
 hidden_3/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0hidden_3_6583486hidden_3_6583488*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_3_layer_call_and_return_conditional_losses_65834852"
 hidden_3/StatefulPartitionedCall?
 hidden_4/StatefulPartitionedCallStatefulPartitionedCall)hidden_3/StatefulPartitionedCall:output:0hidden_4_6583503hidden_4_6583505*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_4_layer_call_and_return_conditional_losses_65835022"
 hidden_4/StatefulPartitionedCall?
 hidden_5/StatefulPartitionedCallStatefulPartitionedCall)hidden_4/StatefulPartitionedCall:output:0hidden_5_6583520hidden_5_6583522*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_5_layer_call_and_return_conditional_losses_65835192"
 hidden_5/StatefulPartitionedCall?
 hidden_6/StatefulPartitionedCallStatefulPartitionedCall)hidden_5/StatefulPartitionedCall:output:0hidden_6_6583537hidden_6_6583539*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_6_layer_call_and_return_conditional_losses_65835362"
 hidden_6/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall)hidden_6/StatefulPartitionedCall:output:0output_6583553output_6583555*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_65835522 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0!^hidden_1/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall!^hidden_3/StatefulPartitionedCall!^hidden_4/StatefulPartitionedCall!^hidden_5/StatefulPartitionedCall!^hidden_6/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2D
 hidden_3/StatefulPartitionedCall hidden_3/StatefulPartitionedCall2D
 hidden_4/StatefulPartitionedCall hidden_4/StatefulPartitionedCall2D
 hidden_5/StatefulPartitionedCall hidden_5/StatefulPartitionedCall2D
 hidden_6/StatefulPartitionedCall hidden_6/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_hidden_1_layer_call_fn_6584096

inputs
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_1_layer_call_and_return_conditional_losses_65834512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?&
?
I__inference_simple_model_layer_call_and_return_conditional_losses_6583837	
input"
hidden_1_6583801:@
hidden_1_6583803:@"
hidden_2_6583806:@ 
hidden_2_6583808: "
hidden_3_6583811: @
hidden_3_6583813:@#
hidden_4_6583816:	@?
hidden_4_6583818:	?$
hidden_5_6583821:
??
hidden_5_6583823:	?#
hidden_6_6583826:	?@
hidden_6_6583828:@ 
output_6583831:@
output_6583833:
identity?? hidden_1/StatefulPartitionedCall? hidden_2/StatefulPartitionedCall? hidden_3/StatefulPartitionedCall? hidden_4/StatefulPartitionedCall? hidden_5/StatefulPartitionedCall? hidden_6/StatefulPartitionedCall?output/StatefulPartitionedCall?
 hidden_1/StatefulPartitionedCallStatefulPartitionedCallinputhidden_1_6583801hidden_1_6583803*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_1_layer_call_and_return_conditional_losses_65834512"
 hidden_1/StatefulPartitionedCall?
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0hidden_2_6583806hidden_2_6583808*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_2_layer_call_and_return_conditional_losses_65834682"
 hidden_2/StatefulPartitionedCall?
 hidden_3/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0hidden_3_6583811hidden_3_6583813*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_3_layer_call_and_return_conditional_losses_65834852"
 hidden_3/StatefulPartitionedCall?
 hidden_4/StatefulPartitionedCallStatefulPartitionedCall)hidden_3/StatefulPartitionedCall:output:0hidden_4_6583816hidden_4_6583818*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_4_layer_call_and_return_conditional_losses_65835022"
 hidden_4/StatefulPartitionedCall?
 hidden_5/StatefulPartitionedCallStatefulPartitionedCall)hidden_4/StatefulPartitionedCall:output:0hidden_5_6583821hidden_5_6583823*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_5_layer_call_and_return_conditional_losses_65835192"
 hidden_5/StatefulPartitionedCall?
 hidden_6/StatefulPartitionedCallStatefulPartitionedCall)hidden_5/StatefulPartitionedCall:output:0hidden_6_6583826hidden_6_6583828*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_hidden_6_layer_call_and_return_conditional_losses_65835362"
 hidden_6/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall)hidden_6/StatefulPartitionedCall:output:0output_6583831output_6583833*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_65835522 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0!^hidden_1/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall!^hidden_3/StatefulPartitionedCall!^hidden_4/StatefulPartitionedCall!^hidden_5/StatefulPartitionedCall!^hidden_6/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2D
 hidden_3/StatefulPartitionedCall hidden_3/StatefulPartitionedCall2D
 hidden_4/StatefulPartitionedCall hidden_4/StatefulPartitionedCall2D
 hidden_5/StatefulPartitionedCall hidden_5/StatefulPartitionedCall2D
 hidden_6/StatefulPartitionedCall hidden_6/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
7
input.
serving_default_input:0?????????:
output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?B
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
	optimizer
	trainable_variables

	variables
regularization_losses
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"?>
_tf_keras_sequential?>{"name": "simple_model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "simple_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}, {"class_name": "Dense", "config": {"name": "hidden_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden_4", "trainable": true, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden_5", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden_6", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 6]}, "float32", "input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "simple_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "hidden_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Dense", "config": {"name": "hidden_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "hidden_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9}, {"class_name": "Dense", "config": {"name": "hidden_4", "trainable": true, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "Dense", "config": {"name": "hidden_5", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15}, {"class_name": "Dense", "config": {"name": "hidden_6", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 21}]}}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.009999999776482582, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "hidden_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "hidden_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "hidden_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "hidden_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 24}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "hidden_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "hidden_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 25}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?

 kernel
!bias
"trainable_variables
#	variables
$regularization_losses
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "hidden_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "hidden_4", "trainable": true, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 26}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

&kernel
'bias
(trainable_variables
)	variables
*regularization_losses
+	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "hidden_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "hidden_5", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 27}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

,kernel
-bias
.trainable_variables
/	variables
0regularization_losses
1	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "hidden_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "hidden_6", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

2kernel
3bias
4trainable_variables
5	variables
6regularization_losses
7	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 29}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
8iter

9beta_1

:beta_2
	;decay
<learning_ratemjmkmlmmmnmo mp!mq&mr'ms,mt-mu2mv3mwvxvyvzv{v|v} v~!v&v?'v?,v?-v?2v?3v?"
	optimizer
?
0
1
2
3
4
5
 6
!7
&8
'9
,10
-11
212
313"
trackable_list_wrapper
?
0
1
2
3
4
5
 6
!7
&8
'9
,10
-11
212
313"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	trainable_variables

=layers

	variables
regularization_losses
>layer_regularization_losses
?non_trainable_variables
@metrics
Alayer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
!:@2hidden_1/kernel
:@2hidden_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables

Blayers
	variables
regularization_losses
Clayer_regularization_losses
Dnon_trainable_variables
Emetrics
Flayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@ 2hidden_2/kernel
: 2hidden_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables

Glayers
	variables
regularization_losses
Hlayer_regularization_losses
Inon_trainable_variables
Jmetrics
Klayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!: @2hidden_3/kernel
:@2hidden_3/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables

Llayers
	variables
regularization_losses
Mlayer_regularization_losses
Nnon_trainable_variables
Ometrics
Player_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	@?2hidden_4/kernel
:?2hidden_4/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
"trainable_variables

Qlayers
#	variables
$regularization_losses
Rlayer_regularization_losses
Snon_trainable_variables
Tmetrics
Ulayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2hidden_5/kernel
:?2hidden_5/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
(trainable_variables

Vlayers
)	variables
*regularization_losses
Wlayer_regularization_losses
Xnon_trainable_variables
Ymetrics
Zlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?@2hidden_6/kernel
:@2hidden_6/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
.trainable_variables

[layers
/	variables
0regularization_losses
\layer_regularization_losses
]non_trainable_variables
^metrics
_layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:@2output/kernel
:2output/bias
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
?
4trainable_variables

`layers
5	variables
6regularization_losses
alayer_regularization_losses
bnon_trainable_variables
cmetrics
dlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
e0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
	ftotal
	gcount
h	variables
i	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 30}
:  (2total
:  (2count
.
f0
g1"
trackable_list_wrapper
-
h	variables"
_generic_user_object
&:$@2Adam/hidden_1/kernel/m
 :@2Adam/hidden_1/bias/m
&:$@ 2Adam/hidden_2/kernel/m
 : 2Adam/hidden_2/bias/m
&:$ @2Adam/hidden_3/kernel/m
 :@2Adam/hidden_3/bias/m
':%	@?2Adam/hidden_4/kernel/m
!:?2Adam/hidden_4/bias/m
(:&
??2Adam/hidden_5/kernel/m
!:?2Adam/hidden_5/bias/m
':%	?@2Adam/hidden_6/kernel/m
 :@2Adam/hidden_6/bias/m
$:"@2Adam/output/kernel/m
:2Adam/output/bias/m
&:$@2Adam/hidden_1/kernel/v
 :@2Adam/hidden_1/bias/v
&:$@ 2Adam/hidden_2/kernel/v
 : 2Adam/hidden_2/bias/v
&:$ @2Adam/hidden_3/kernel/v
 :@2Adam/hidden_3/bias/v
':%	@?2Adam/hidden_4/kernel/v
!:?2Adam/hidden_4/bias/v
(:&
??2Adam/hidden_5/kernel/v
!:?2Adam/hidden_5/bias/v
':%	?@2Adam/hidden_6/kernel/v
 :@2Adam/hidden_6/bias/v
$:"@2Adam/output/kernel/v
:2Adam/output/bias/v
?2?
"__inference__wrapped_model_6583433?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *$?!
?
input?????????
?2?
.__inference_simple_model_layer_call_fn_6583590
.__inference_simple_model_layer_call_fn_6583950
.__inference_simple_model_layer_call_fn_6583983
.__inference_simple_model_layer_call_fn_6583798?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_simple_model_layer_call_and_return_conditional_losses_6584035
I__inference_simple_model_layer_call_and_return_conditional_losses_6584087
I__inference_simple_model_layer_call_and_return_conditional_losses_6583837
I__inference_simple_model_layer_call_and_return_conditional_losses_6583876?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_hidden_1_layer_call_fn_6584096?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_hidden_1_layer_call_and_return_conditional_losses_6584107?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_hidden_2_layer_call_fn_6584116?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_hidden_2_layer_call_and_return_conditional_losses_6584127?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_hidden_3_layer_call_fn_6584136?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_hidden_3_layer_call_and_return_conditional_losses_6584147?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_hidden_4_layer_call_fn_6584156?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_hidden_4_layer_call_and_return_conditional_losses_6584167?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_hidden_5_layer_call_fn_6584176?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_hidden_5_layer_call_and_return_conditional_losses_6584187?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_hidden_6_layer_call_fn_6584196?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_hidden_6_layer_call_and_return_conditional_losses_6584207?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_output_layer_call_fn_6584216?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_output_layer_call_and_return_conditional_losses_6584226?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_6583917input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_6583433q !&',-23.?+
$?!
?
input?????????
? "/?,
*
output ?
output??????????
E__inference_hidden_1_layer_call_and_return_conditional_losses_6584107\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????@
? }
*__inference_hidden_1_layer_call_fn_6584096O/?,
%?"
 ?
inputs?????????
? "??????????@?
E__inference_hidden_2_layer_call_and_return_conditional_losses_6584127\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0????????? 
? }
*__inference_hidden_2_layer_call_fn_6584116O/?,
%?"
 ?
inputs?????????@
? "?????????? ?
E__inference_hidden_3_layer_call_and_return_conditional_losses_6584147\/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????@
? }
*__inference_hidden_3_layer_call_fn_6584136O/?,
%?"
 ?
inputs????????? 
? "??????????@?
E__inference_hidden_4_layer_call_and_return_conditional_losses_6584167] !/?,
%?"
 ?
inputs?????????@
? "&?#
?
0??????????
? ~
*__inference_hidden_4_layer_call_fn_6584156P !/?,
%?"
 ?
inputs?????????@
? "????????????
E__inference_hidden_5_layer_call_and_return_conditional_losses_6584187^&'0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_hidden_5_layer_call_fn_6584176Q&'0?-
&?#
!?
inputs??????????
? "????????????
E__inference_hidden_6_layer_call_and_return_conditional_losses_6584207],-0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? ~
*__inference_hidden_6_layer_call_fn_6584196P,-0?-
&?#
!?
inputs??????????
? "??????????@?
C__inference_output_layer_call_and_return_conditional_losses_6584226\23/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? {
(__inference_output_layer_call_fn_6584216O23/?,
%?"
 ?
inputs?????????@
? "???????????
%__inference_signature_wrapper_6583917z !&',-237?4
? 
-?*
(
input?
input?????????"/?,
*
output ?
output??????????
I__inference_simple_model_layer_call_and_return_conditional_losses_6583837o !&',-236?3
,?)
?
input?????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_simple_model_layer_call_and_return_conditional_losses_6583876o !&',-236?3
,?)
?
input?????????
p

 
? "%?"
?
0?????????
? ?
I__inference_simple_model_layer_call_and_return_conditional_losses_6584035p !&',-237?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_simple_model_layer_call_and_return_conditional_losses_6584087p !&',-237?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
.__inference_simple_model_layer_call_fn_6583590b !&',-236?3
,?)
?
input?????????
p 

 
? "???????????
.__inference_simple_model_layer_call_fn_6583798b !&',-236?3
,?)
?
input?????????
p

 
? "???????????
.__inference_simple_model_layer_call_fn_6583950c !&',-237?4
-?*
 ?
inputs?????????
p 

 
? "???????????
.__inference_simple_model_layer_call_fn_6583983c !&',-237?4
-?*
 ?
inputs?????????
p

 
? "??????????
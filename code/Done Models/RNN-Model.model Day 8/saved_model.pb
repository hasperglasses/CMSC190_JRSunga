гд"
™э
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
Њ
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
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8ЂЂ 
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:@*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
Р
lstm_2/lstm_cell_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ЦА**
shared_namelstm_2/lstm_cell_2/kernel
Й
-lstm_2/lstm_cell_2/kernel/Read/ReadVariableOpReadVariableOplstm_2/lstm_cell_2/kernel* 
_output_shapes
:
ЦА*
dtype0
£
#lstm_2/lstm_cell_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*4
shared_name%#lstm_2/lstm_cell_2/recurrent_kernel
Ь
7lstm_2/lstm_cell_2/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_2/lstm_cell_2/recurrent_kernel*
_output_shapes
:	@А*
dtype0
З
lstm_2/lstm_cell_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_namelstm_2/lstm_cell_2/bias
А
+lstm_2/lstm_cell_2/bias/Read/ReadVariableOpReadVariableOplstm_2/lstm_cell_2/bias*
_output_shapes	
:А*
dtype0
П
lstm_3/lstm_cell_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А**
shared_namelstm_3/lstm_cell_3/kernel
И
-lstm_3/lstm_cell_3/kernel/Read/ReadVariableOpReadVariableOplstm_3/lstm_cell_3/kernel*
_output_shapes
:	@А*
dtype0
£
#lstm_3/lstm_cell_3/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*4
shared_name%#lstm_3/lstm_cell_3/recurrent_kernel
Ь
7lstm_3/lstm_cell_3/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_3/lstm_cell_3/recurrent_kernel*
_output_shapes
:	@А*
dtype0
З
lstm_3/lstm_cell_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_namelstm_3/lstm_cell_3/bias
А
+lstm_3/lstm_cell_3/bias/Read/ReadVariableOpReadVariableOplstm_3/lstm_cell_3/bias*
_output_shapes	
:А*
dtype0

NoOpNoOp
…$
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Д$
valueъ#Bч# Bр#
і
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
	optimizer
	
signatures

trainable_variables
regularization_losses
	variables
	keras_api
l
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
l
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
R
	variables
regularization_losses
 trainable_variables
!	keras_api
h

"kernel
#bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
R
(	variables
)regularization_losses
*trainable_variables
+	keras_api
h

,kernel
-bias
.	variables
/regularization_losses
0trainable_variables
1	keras_api
 
 
F
20
31
42
53
64
75
"6
#7
,8
-9
 
F
20
31
42
53
64
75
"6
#7
,8
-9
≠
8non_trainable_variables

trainable_variables

9layers
regularization_losses
:metrics
;layer_metrics
	variables
<layer_regularization_losses
~

2kernel
3recurrent_kernel
4bias
=	variables
>regularization_losses
?trainable_variables
@	keras_api
 

20
31
42

20
31
42
 
є
Anon_trainable_variables
trainable_variables

Blayers
	variables
regularization_losses
Cmetrics
Dlayer_metrics

Estates
Flayer_regularization_losses
 
 
 
≠
Gnon_trainable_variables

Hlayers
	variables
regularization_losses
Imetrics
Jlayer_metrics
trainable_variables
Klayer_regularization_losses
~

5kernel
6recurrent_kernel
7bias
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
 

50
61
72

50
61
72
 
є
Pnon_trainable_variables
trainable_variables

Qlayers
	variables
regularization_losses
Rmetrics
Slayer_metrics

Tstates
Ulayer_regularization_losses
 
 
 
≠
Vnon_trainable_variables

Wlayers
	variables
regularization_losses
Xmetrics
Ylayer_metrics
 trainable_variables
Zlayer_regularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
≠
[non_trainable_variables

\layers
$	variables
%regularization_losses
]metrics
^layer_metrics
&trainable_variables
_layer_regularization_losses
 
 
 
≠
`non_trainable_variables

alayers
(	variables
)regularization_losses
bmetrics
clayer_metrics
*trainable_variables
dlayer_regularization_losses
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1
 

,0
-1
≠
enon_trainable_variables

flayers
.	variables
/regularization_losses
gmetrics
hlayer_metrics
0trainable_variables
ilayer_regularization_losses
_]
VARIABLE_VALUElstm_2/lstm_cell_2/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#lstm_2/lstm_cell_2/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUElstm_2/lstm_cell_2/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_3/lstm_cell_3/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#lstm_3/lstm_cell_3/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUElstm_3/lstm_cell_3/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
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
 

20
31
42
 

20
31
42
≠
jnon_trainable_variables

klayers
=	variables
>regularization_losses
lmetrics
mlayer_metrics
?trainable_variables
nlayer_regularization_losses
 

0
 
 
 
 
 
 
 
 
 

50
61
72
 

50
61
72
≠
onon_trainable_variables

players
L	variables
Mregularization_losses
qmetrics
rlayer_metrics
Ntrainable_variables
slayer_regularization_losses
 

0
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
Ж
serving_default_input_5Placeholder*-
_output_shapes
:€€€€€€€€€ЦЦ*
dtype0*"
shape:€€€€€€€€€ЦЦ
С
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5lstm_2/lstm_cell_2/kernel#lstm_2/lstm_cell_2/recurrent_kernellstm_2/lstm_cell_2/biaslstm_3/lstm_cell_3/kernel#lstm_3/lstm_cell_3/recurrent_kernellstm_3/lstm_cell_3/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*,
_read_only_resource_inputs

	
**
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference_signature_wrapper_46852
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Є
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp-lstm_2/lstm_cell_2/kernel/Read/ReadVariableOp7lstm_2/lstm_cell_2/recurrent_kernel/Read/ReadVariableOp+lstm_2/lstm_cell_2/bias/Read/ReadVariableOp-lstm_3/lstm_cell_3/kernel/Read/ReadVariableOp7lstm_3/lstm_cell_3/recurrent_kernel/Read/ReadVariableOp+lstm_3/lstm_cell_3/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*'
f"R 
__inference__traced_save_49251
л
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/biasdense_3/kerneldense_3/biaslstm_2/lstm_cell_2/kernel#lstm_2/lstm_cell_2/recurrent_kernellstm_2/lstm_cell_2/biaslstm_3/lstm_cell_3/kernel#lstm_3/lstm_cell_3/recurrent_kernellstm_3/lstm_cell_3/bias*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__traced_restore_49293їв
Љ	
Ц
$sequential_1_lstm_2_while_cond_44390*
&sequential_1_lstm_2_while_loop_counter0
,sequential_1_lstm_2_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3,
(less_sequential_1_lstm_2_strided_slice_1A
=sequential_1_lstm_2_while_cond_44390___redundant_placeholder0A
=sequential_1_lstm_2_while_cond_44390___redundant_placeholder1A
=sequential_1_lstm_2_while_cond_44390___redundant_placeholder2A
=sequential_1_lstm_2_while_cond_44390___redundant_placeholder3
identity
l
LessLessplaceholder(less_sequential_1_lstm_2_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
л
™
B__inference_dense_3_layer_call_and_return_conditional_losses_46664

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€:::O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
д2
п
while_body_47957
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
,lstm_cell_4_matmul_readvariableop_resource_02
.lstm_cell_4_matmul_1_readvariableop_resource_01
-lstm_cell_4_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resourceИЈ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Ц   23
1TensorArrayV2Read/TensorListGetItem/element_shapeґ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€Ц*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemµ
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_readvariableop_resource_0* 
_output_shapes
:
ЦА*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOpЉ
lstm_cell_4/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMulЇ
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp•
lstm_cell_4/MatMul_1MatMulplaceholder_2+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMul_1Ь
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/add≥
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp©
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimп
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_4/splitГ
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/SigmoidЗ
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_1Е
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/ReluШ
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_1Н
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_1З
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_2Ѕ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3m

Identity_4Identitylstm_cell_4/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_4m

Identity_5Identitylstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"\
+lstm_cell_4_biasadd_readvariableop_resource-lstm_cell_4_biasadd_readvariableop_resource_0"^
,lstm_cell_4_matmul_1_readvariableop_resource.lstm_cell_4_matmul_1_readvariableop_resource_0"Z
*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :€€€€€€€€€@:€€€€€€€€€@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
∆3
†
lstm_2_while_body_46920
lstm_2_while_loop_counter#
lstm_2_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_2_strided_slice_1_0X
Ttensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_00
,lstm_cell_4_matmul_readvariableop_resource_02
.lstm_cell_4_matmul_1_readvariableop_resource_01
-lstm_cell_4_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_2_strided_slice_1V
Rtensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resourceИЈ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Ц   23
1TensorArrayV2Read/TensorListGetItem/element_shapeљ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemTtensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€Ц*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemµ
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_readvariableop_resource_0* 
_output_shapes
:
ЦА*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOpЉ
lstm_cell_4/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMulЇ
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp•
lstm_cell_4/MatMul_1MatMulplaceholder_2+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMul_1Ь
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/add≥
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp©
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimп
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_4/splitГ
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/SigmoidЗ
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_1Е
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/ReluШ
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_1Н
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_1З
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_2Ѕ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/ye
add_1AddV2lstm_2_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identityf

Identity_1Identitylstm_2_while_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3m

Identity_4Identitylstm_cell_4/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_4m

Identity_5Identitylstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"2
lstm_2_strided_slice_1lstm_2_strided_slice_1_0"\
+lstm_cell_4_biasadd_readvariableop_resource-lstm_cell_4_biasadd_readvariableop_resource_0"^
,lstm_cell_4_matmul_1_readvariableop_resource.lstm_cell_4_matmul_1_readvariableop_resource_0"Z
*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_readvariableop_resource_0"™
Rtensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensorTtensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :€€€€€€€€€@:€€€€€€€€€@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
в2
п
while_body_48793
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
,lstm_cell_5_matmul_readvariableop_resource_02
.lstm_cell_5_matmul_1_readvariableop_resource_01
-lstm_cell_5_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resourceИЈ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   23
1TensorArrayV2Read/TensorListGetItem/element_shapeµ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€@*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemі
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOpЉ
lstm_cell_5/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMulЇ
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp•
lstm_cell_5/MatMul_1MatMulplaceholder_2+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMul_1Ь
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/add≥
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp©
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dimп
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_5/splitГ
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/SigmoidЗ
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_1Е
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/ReluШ
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_1Н
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/add_1З
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Relu_1Ь
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_2Ѕ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3m

Identity_4Identitylstm_cell_5/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_4m

Identity_5Identitylstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"\
+lstm_cell_5_biasadd_readvariableop_resource-lstm_cell_5_biasadd_readvariableop_resource_0"^
,lstm_cell_5_matmul_1_readvariableop_resource.lstm_cell_5_matmul_1_readvariableop_resource_0"Z
*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :€€€€€€€€€@:€€€€€€€€€@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
Ж3
Д
!__inference__traced_restore_49293
file_prefix#
assignvariableop_dense_2_kernel#
assignvariableop_1_dense_2_bias%
!assignvariableop_2_dense_3_kernel#
assignvariableop_3_dense_3_bias0
,assignvariableop_4_lstm_2_lstm_cell_2_kernel:
6assignvariableop_5_lstm_2_lstm_cell_2_recurrent_kernel.
*assignvariableop_6_lstm_2_lstm_cell_2_bias0
,assignvariableop_7_lstm_3_lstm_cell_3_kernel:
6assignvariableop_8_lstm_3_lstm_cell_3_recurrent_kernel.
*assignvariableop_9_lstm_3_lstm_cell_3_bias
identity_11ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_2ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Ґ	RestoreV2ҐRestoreV2_1С
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*Э
valueУBР
B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesҐ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B 2
RestoreV2/shape_and_slicesЁ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*<
_output_shapes*
(::::::::::*
dtypes
2
2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityП
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Х
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2Ч
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_3_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3Х
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_3_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4Ґ
AssignVariableOp_4AssignVariableOp,assignvariableop_4_lstm_2_lstm_cell_2_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5ђ
AssignVariableOp_5AssignVariableOp6assignvariableop_5_lstm_2_lstm_cell_2_recurrent_kernelIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6†
AssignVariableOp_6AssignVariableOp*assignvariableop_6_lstm_2_lstm_cell_2_biasIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Ґ
AssignVariableOp_7AssignVariableOp,assignvariableop_7_lstm_3_lstm_cell_3_kernelIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8ђ
AssignVariableOp_8AssignVariableOp6assignvariableop_8_lstm_3_lstm_cell_3_recurrent_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9†
AssignVariableOp_9AssignVariableOp*assignvariableop_9_lstm_3_lstm_cell_3_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9®
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesƒ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЇ
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10«
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_11"#
identity_11Identity_11:output:0*=
_input_shapes,
*: ::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
ОМ
ь
 __inference__wrapped_model_44642
input_5B
>sequential_1_lstm_2_lstm_cell_4_matmul_readvariableop_resourceD
@sequential_1_lstm_2_lstm_cell_4_matmul_1_readvariableop_resourceC
?sequential_1_lstm_2_lstm_cell_4_biasadd_readvariableop_resourceB
>sequential_1_lstm_3_lstm_cell_5_matmul_readvariableop_resourceD
@sequential_1_lstm_3_lstm_cell_5_matmul_1_readvariableop_resourceC
?sequential_1_lstm_3_lstm_cell_5_biasadd_readvariableop_resource7
3sequential_1_dense_2_matmul_readvariableop_resource8
4sequential_1_dense_2_biasadd_readvariableop_resource7
3sequential_1_dense_3_matmul_readvariableop_resource8
4sequential_1_dense_3_biasadd_readvariableop_resource
identityИҐsequential_1/lstm_2/whileҐsequential_1/lstm_3/whilem
sequential_1/lstm_2/ShapeShapeinput_5*
T0*
_output_shapes
:2
sequential_1/lstm_2/ShapeЬ
'sequential_1/lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_1/lstm_2/strided_slice/stack†
)sequential_1/lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_1/lstm_2/strided_slice/stack_1†
)sequential_1/lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_1/lstm_2/strided_slice/stack_2Џ
!sequential_1/lstm_2/strided_sliceStridedSlice"sequential_1/lstm_2/Shape:output:00sequential_1/lstm_2/strided_slice/stack:output:02sequential_1/lstm_2/strided_slice/stack_1:output:02sequential_1/lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_1/lstm_2/strided_sliceД
sequential_1/lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2!
sequential_1/lstm_2/zeros/mul/yЉ
sequential_1/lstm_2/zeros/mulMul*sequential_1/lstm_2/strided_slice:output:0(sequential_1/lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_1/lstm_2/zeros/mulЗ
 sequential_1/lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2"
 sequential_1/lstm_2/zeros/Less/yЈ
sequential_1/lstm_2/zeros/LessLess!sequential_1/lstm_2/zeros/mul:z:0)sequential_1/lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_1/lstm_2/zeros/LessК
"sequential_1/lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2$
"sequential_1/lstm_2/zeros/packed/1”
 sequential_1/lstm_2/zeros/packedPack*sequential_1/lstm_2/strided_slice:output:0+sequential_1/lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_1/lstm_2/zeros/packedЗ
sequential_1/lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_1/lstm_2/zeros/Const≈
sequential_1/lstm_2/zerosFill)sequential_1/lstm_2/zeros/packed:output:0(sequential_1/lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
sequential_1/lstm_2/zerosИ
!sequential_1/lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2#
!sequential_1/lstm_2/zeros_1/mul/y¬
sequential_1/lstm_2/zeros_1/mulMul*sequential_1/lstm_2/strided_slice:output:0*sequential_1/lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_1/lstm_2/zeros_1/mulЛ
"sequential_1/lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2$
"sequential_1/lstm_2/zeros_1/Less/yњ
 sequential_1/lstm_2/zeros_1/LessLess#sequential_1/lstm_2/zeros_1/mul:z:0+sequential_1/lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_1/lstm_2/zeros_1/LessО
$sequential_1/lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2&
$sequential_1/lstm_2/zeros_1/packed/1ў
"sequential_1/lstm_2/zeros_1/packedPack*sequential_1/lstm_2/strided_slice:output:0-sequential_1/lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/lstm_2/zeros_1/packedЛ
!sequential_1/lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_1/lstm_2/zeros_1/ConstЌ
sequential_1/lstm_2/zeros_1Fill+sequential_1/lstm_2/zeros_1/packed:output:0*sequential_1/lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
sequential_1/lstm_2/zeros_1Э
"sequential_1/lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_1/lstm_2/transpose/permє
sequential_1/lstm_2/transpose	Transposeinput_5+sequential_1/lstm_2/transpose/perm:output:0*
T0*-
_output_shapes
:Ц€€€€€€€€€Ц2
sequential_1/lstm_2/transposeЛ
sequential_1/lstm_2/Shape_1Shape!sequential_1/lstm_2/transpose:y:0*
T0*
_output_shapes
:2
sequential_1/lstm_2/Shape_1†
)sequential_1/lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_1/lstm_2/strided_slice_1/stack§
+sequential_1/lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_2/strided_slice_1/stack_1§
+sequential_1/lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_2/strided_slice_1/stack_2ж
#sequential_1/lstm_2/strided_slice_1StridedSlice$sequential_1/lstm_2/Shape_1:output:02sequential_1/lstm_2/strided_slice_1/stack:output:04sequential_1/lstm_2/strided_slice_1/stack_1:output:04sequential_1/lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_1/lstm_2/strided_slice_1≠
/sequential_1/lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€21
/sequential_1/lstm_2/TensorArrayV2/element_shapeВ
!sequential_1/lstm_2/TensorArrayV2TensorListReserve8sequential_1/lstm_2/TensorArrayV2/element_shape:output:0,sequential_1/lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_1/lstm_2/TensorArrayV2з
Isequential_1/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Ц   2K
Isequential_1/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape»
;sequential_1/lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_1/lstm_2/transpose:y:0Rsequential_1/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_1/lstm_2/TensorArrayUnstack/TensorListFromTensor†
)sequential_1/lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_1/lstm_2/strided_slice_2/stack§
+sequential_1/lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_2/strided_slice_2/stack_1§
+sequential_1/lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_2/strided_slice_2/stack_2х
#sequential_1/lstm_2/strided_slice_2StridedSlice!sequential_1/lstm_2/transpose:y:02sequential_1/lstm_2/strided_slice_2/stack:output:04sequential_1/lstm_2/strided_slice_2/stack_1:output:04sequential_1/lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€Ц*
shrink_axis_mask2%
#sequential_1/lstm_2/strided_slice_2п
5sequential_1/lstm_2/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp>sequential_1_lstm_2_lstm_cell_4_matmul_readvariableop_resource* 
_output_shapes
:
ЦА*
dtype027
5sequential_1/lstm_2/lstm_cell_4/MatMul/ReadVariableOpъ
&sequential_1/lstm_2/lstm_cell_4/MatMulMatMul,sequential_1/lstm_2/strided_slice_2:output:0=sequential_1/lstm_2/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2(
&sequential_1/lstm_2/lstm_cell_4/MatMulф
7sequential_1/lstm_2/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp@sequential_1_lstm_2_lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype029
7sequential_1/lstm_2/lstm_cell_4/MatMul_1/ReadVariableOpц
(sequential_1/lstm_2/lstm_cell_4/MatMul_1MatMul"sequential_1/lstm_2/zeros:output:0?sequential_1/lstm_2/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2*
(sequential_1/lstm_2/lstm_cell_4/MatMul_1м
#sequential_1/lstm_2/lstm_cell_4/addAddV20sequential_1/lstm_2/lstm_cell_4/MatMul:product:02sequential_1/lstm_2/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2%
#sequential_1/lstm_2/lstm_cell_4/addн
6sequential_1/lstm_2/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_lstm_2_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype028
6sequential_1/lstm_2/lstm_cell_4/BiasAdd/ReadVariableOpщ
'sequential_1/lstm_2/lstm_cell_4/BiasAddBiasAdd'sequential_1/lstm_2/lstm_cell_4/add:z:0>sequential_1/lstm_2/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2)
'sequential_1/lstm_2/lstm_cell_4/BiasAddР
%sequential_1/lstm_2/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/lstm_2/lstm_cell_4/Const§
/sequential_1/lstm_2/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential_1/lstm_2/lstm_cell_4/split/split_dimњ
%sequential_1/lstm_2/lstm_cell_4/splitSplit8sequential_1/lstm_2/lstm_cell_4/split/split_dim:output:00sequential_1/lstm_2/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2'
%sequential_1/lstm_2/lstm_cell_4/splitњ
'sequential_1/lstm_2/lstm_cell_4/SigmoidSigmoid.sequential_1/lstm_2/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2)
'sequential_1/lstm_2/lstm_cell_4/Sigmoid√
)sequential_1/lstm_2/lstm_cell_4/Sigmoid_1Sigmoid.sequential_1/lstm_2/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2+
)sequential_1/lstm_2/lstm_cell_4/Sigmoid_1Ў
#sequential_1/lstm_2/lstm_cell_4/mulMul-sequential_1/lstm_2/lstm_cell_4/Sigmoid_1:y:0$sequential_1/lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2%
#sequential_1/lstm_2/lstm_cell_4/mulґ
$sequential_1/lstm_2/lstm_cell_4/ReluRelu.sequential_1/lstm_2/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2&
$sequential_1/lstm_2/lstm_cell_4/Reluи
%sequential_1/lstm_2/lstm_cell_4/mul_1Mul+sequential_1/lstm_2/lstm_cell_4/Sigmoid:y:02sequential_1/lstm_2/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%sequential_1/lstm_2/lstm_cell_4/mul_1Ё
%sequential_1/lstm_2/lstm_cell_4/add_1AddV2'sequential_1/lstm_2/lstm_cell_4/mul:z:0)sequential_1/lstm_2/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%sequential_1/lstm_2/lstm_cell_4/add_1√
)sequential_1/lstm_2/lstm_cell_4/Sigmoid_2Sigmoid.sequential_1/lstm_2/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2+
)sequential_1/lstm_2/lstm_cell_4/Sigmoid_2µ
&sequential_1/lstm_2/lstm_cell_4/Relu_1Relu)sequential_1/lstm_2/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2(
&sequential_1/lstm_2/lstm_cell_4/Relu_1м
%sequential_1/lstm_2/lstm_cell_4/mul_2Mul-sequential_1/lstm_2/lstm_cell_4/Sigmoid_2:y:04sequential_1/lstm_2/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%sequential_1/lstm_2/lstm_cell_4/mul_2Ј
1sequential_1/lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   23
1sequential_1/lstm_2/TensorArrayV2_1/element_shapeИ
#sequential_1/lstm_2/TensorArrayV2_1TensorListReserve:sequential_1/lstm_2/TensorArrayV2_1/element_shape:output:0,sequential_1/lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_1/lstm_2/TensorArrayV2_1v
sequential_1/lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_1/lstm_2/timeІ
,sequential_1/lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2.
,sequential_1/lstm_2/while/maximum_iterationsТ
&sequential_1/lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_1/lstm_2/while/loop_counterХ
sequential_1/lstm_2/whileWhile/sequential_1/lstm_2/while/loop_counter:output:05sequential_1/lstm_2/while/maximum_iterations:output:0!sequential_1/lstm_2/time:output:0,sequential_1/lstm_2/TensorArrayV2_1:handle:0"sequential_1/lstm_2/zeros:output:0$sequential_1/lstm_2/zeros_1:output:0,sequential_1/lstm_2/strided_slice_1:output:0Ksequential_1/lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_1_lstm_2_lstm_cell_4_matmul_readvariableop_resource@sequential_1_lstm_2_lstm_cell_4_matmul_1_readvariableop_resource?sequential_1_lstm_2_lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*0
body(R&
$sequential_1_lstm_2_while_body_44391*0
cond(R&
$sequential_1_lstm_2_while_cond_44390*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
sequential_1/lstm_2/whileЁ
Dsequential_1/lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2F
Dsequential_1/lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeє
6sequential_1/lstm_2/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_1/lstm_2/while:output:3Msequential_1/lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:Ц€€€€€€€€€@*
element_dtype028
6sequential_1/lstm_2/TensorArrayV2Stack/TensorListStack©
)sequential_1/lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2+
)sequential_1/lstm_2/strided_slice_3/stack§
+sequential_1/lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_1/lstm_2/strided_slice_3/stack_1§
+sequential_1/lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_2/strided_slice_3/stack_2Т
#sequential_1/lstm_2/strided_slice_3StridedSlice?sequential_1/lstm_2/TensorArrayV2Stack/TensorListStack:tensor:02sequential_1/lstm_2/strided_slice_3/stack:output:04sequential_1/lstm_2/strided_slice_3/stack_1:output:04sequential_1/lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2%
#sequential_1/lstm_2/strided_slice_3°
$sequential_1/lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_1/lstm_2/transpose_1/permц
sequential_1/lstm_2/transpose_1	Transpose?sequential_1/lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_1/lstm_2/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2!
sequential_1/lstm_2/transpose_1О
sequential_1/lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_1/lstm_2/runtime™
sequential_1/dropout_3/IdentityIdentity#sequential_1/lstm_2/transpose_1:y:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2!
sequential_1/dropout_3/IdentityО
sequential_1/lstm_3/ShapeShape(sequential_1/dropout_3/Identity:output:0*
T0*
_output_shapes
:2
sequential_1/lstm_3/ShapeЬ
'sequential_1/lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_1/lstm_3/strided_slice/stack†
)sequential_1/lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_1/lstm_3/strided_slice/stack_1†
)sequential_1/lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_1/lstm_3/strided_slice/stack_2Џ
!sequential_1/lstm_3/strided_sliceStridedSlice"sequential_1/lstm_3/Shape:output:00sequential_1/lstm_3/strided_slice/stack:output:02sequential_1/lstm_3/strided_slice/stack_1:output:02sequential_1/lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_1/lstm_3/strided_sliceД
sequential_1/lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2!
sequential_1/lstm_3/zeros/mul/yЉ
sequential_1/lstm_3/zeros/mulMul*sequential_1/lstm_3/strided_slice:output:0(sequential_1/lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_1/lstm_3/zeros/mulЗ
 sequential_1/lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2"
 sequential_1/lstm_3/zeros/Less/yЈ
sequential_1/lstm_3/zeros/LessLess!sequential_1/lstm_3/zeros/mul:z:0)sequential_1/lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_1/lstm_3/zeros/LessК
"sequential_1/lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2$
"sequential_1/lstm_3/zeros/packed/1”
 sequential_1/lstm_3/zeros/packedPack*sequential_1/lstm_3/strided_slice:output:0+sequential_1/lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_1/lstm_3/zeros/packedЗ
sequential_1/lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_1/lstm_3/zeros/Const≈
sequential_1/lstm_3/zerosFill)sequential_1/lstm_3/zeros/packed:output:0(sequential_1/lstm_3/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
sequential_1/lstm_3/zerosИ
!sequential_1/lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2#
!sequential_1/lstm_3/zeros_1/mul/y¬
sequential_1/lstm_3/zeros_1/mulMul*sequential_1/lstm_3/strided_slice:output:0*sequential_1/lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_1/lstm_3/zeros_1/mulЛ
"sequential_1/lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2$
"sequential_1/lstm_3/zeros_1/Less/yњ
 sequential_1/lstm_3/zeros_1/LessLess#sequential_1/lstm_3/zeros_1/mul:z:0+sequential_1/lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_1/lstm_3/zeros_1/LessО
$sequential_1/lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2&
$sequential_1/lstm_3/zeros_1/packed/1ў
"sequential_1/lstm_3/zeros_1/packedPack*sequential_1/lstm_3/strided_slice:output:0-sequential_1/lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/lstm_3/zeros_1/packedЛ
!sequential_1/lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_1/lstm_3/zeros_1/ConstЌ
sequential_1/lstm_3/zeros_1Fill+sequential_1/lstm_3/zeros_1/packed:output:0*sequential_1/lstm_3/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
sequential_1/lstm_3/zeros_1Э
"sequential_1/lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_1/lstm_3/transpose/permў
sequential_1/lstm_3/transpose	Transpose(sequential_1/dropout_3/Identity:output:0+sequential_1/lstm_3/transpose/perm:output:0*
T0*,
_output_shapes
:Ц€€€€€€€€€@2
sequential_1/lstm_3/transposeЛ
sequential_1/lstm_3/Shape_1Shape!sequential_1/lstm_3/transpose:y:0*
T0*
_output_shapes
:2
sequential_1/lstm_3/Shape_1†
)sequential_1/lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_1/lstm_3/strided_slice_1/stack§
+sequential_1/lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_3/strided_slice_1/stack_1§
+sequential_1/lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_3/strided_slice_1/stack_2ж
#sequential_1/lstm_3/strided_slice_1StridedSlice$sequential_1/lstm_3/Shape_1:output:02sequential_1/lstm_3/strided_slice_1/stack:output:04sequential_1/lstm_3/strided_slice_1/stack_1:output:04sequential_1/lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_1/lstm_3/strided_slice_1≠
/sequential_1/lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€21
/sequential_1/lstm_3/TensorArrayV2/element_shapeВ
!sequential_1/lstm_3/TensorArrayV2TensorListReserve8sequential_1/lstm_3/TensorArrayV2/element_shape:output:0,sequential_1/lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_1/lstm_3/TensorArrayV2з
Isequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2K
Isequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape»
;sequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_1/lstm_3/transpose:y:0Rsequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor†
)sequential_1/lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_1/lstm_3/strided_slice_2/stack§
+sequential_1/lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_3/strided_slice_2/stack_1§
+sequential_1/lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_3/strided_slice_2/stack_2ф
#sequential_1/lstm_3/strided_slice_2StridedSlice!sequential_1/lstm_3/transpose:y:02sequential_1/lstm_3/strided_slice_2/stack:output:04sequential_1/lstm_3/strided_slice_2/stack_1:output:04sequential_1/lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2%
#sequential_1/lstm_3/strided_slice_2о
5sequential_1/lstm_3/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp>sequential_1_lstm_3_lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype027
5sequential_1/lstm_3/lstm_cell_5/MatMul/ReadVariableOpъ
&sequential_1/lstm_3/lstm_cell_5/MatMulMatMul,sequential_1/lstm_3/strided_slice_2:output:0=sequential_1/lstm_3/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2(
&sequential_1/lstm_3/lstm_cell_5/MatMulф
7sequential_1/lstm_3/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp@sequential_1_lstm_3_lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype029
7sequential_1/lstm_3/lstm_cell_5/MatMul_1/ReadVariableOpц
(sequential_1/lstm_3/lstm_cell_5/MatMul_1MatMul"sequential_1/lstm_3/zeros:output:0?sequential_1/lstm_3/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2*
(sequential_1/lstm_3/lstm_cell_5/MatMul_1м
#sequential_1/lstm_3/lstm_cell_5/addAddV20sequential_1/lstm_3/lstm_cell_5/MatMul:product:02sequential_1/lstm_3/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2%
#sequential_1/lstm_3/lstm_cell_5/addн
6sequential_1/lstm_3/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_lstm_3_lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype028
6sequential_1/lstm_3/lstm_cell_5/BiasAdd/ReadVariableOpщ
'sequential_1/lstm_3/lstm_cell_5/BiasAddBiasAdd'sequential_1/lstm_3/lstm_cell_5/add:z:0>sequential_1/lstm_3/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2)
'sequential_1/lstm_3/lstm_cell_5/BiasAddР
%sequential_1/lstm_3/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/lstm_3/lstm_cell_5/Const§
/sequential_1/lstm_3/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential_1/lstm_3/lstm_cell_5/split/split_dimњ
%sequential_1/lstm_3/lstm_cell_5/splitSplit8sequential_1/lstm_3/lstm_cell_5/split/split_dim:output:00sequential_1/lstm_3/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2'
%sequential_1/lstm_3/lstm_cell_5/splitњ
'sequential_1/lstm_3/lstm_cell_5/SigmoidSigmoid.sequential_1/lstm_3/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2)
'sequential_1/lstm_3/lstm_cell_5/Sigmoid√
)sequential_1/lstm_3/lstm_cell_5/Sigmoid_1Sigmoid.sequential_1/lstm_3/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2+
)sequential_1/lstm_3/lstm_cell_5/Sigmoid_1Ў
#sequential_1/lstm_3/lstm_cell_5/mulMul-sequential_1/lstm_3/lstm_cell_5/Sigmoid_1:y:0$sequential_1/lstm_3/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2%
#sequential_1/lstm_3/lstm_cell_5/mulґ
$sequential_1/lstm_3/lstm_cell_5/ReluRelu.sequential_1/lstm_3/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2&
$sequential_1/lstm_3/lstm_cell_5/Reluи
%sequential_1/lstm_3/lstm_cell_5/mul_1Mul+sequential_1/lstm_3/lstm_cell_5/Sigmoid:y:02sequential_1/lstm_3/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%sequential_1/lstm_3/lstm_cell_5/mul_1Ё
%sequential_1/lstm_3/lstm_cell_5/add_1AddV2'sequential_1/lstm_3/lstm_cell_5/mul:z:0)sequential_1/lstm_3/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%sequential_1/lstm_3/lstm_cell_5/add_1√
)sequential_1/lstm_3/lstm_cell_5/Sigmoid_2Sigmoid.sequential_1/lstm_3/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2+
)sequential_1/lstm_3/lstm_cell_5/Sigmoid_2µ
&sequential_1/lstm_3/lstm_cell_5/Relu_1Relu)sequential_1/lstm_3/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2(
&sequential_1/lstm_3/lstm_cell_5/Relu_1м
%sequential_1/lstm_3/lstm_cell_5/mul_2Mul-sequential_1/lstm_3/lstm_cell_5/Sigmoid_2:y:04sequential_1/lstm_3/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%sequential_1/lstm_3/lstm_cell_5/mul_2Ј
1sequential_1/lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   23
1sequential_1/lstm_3/TensorArrayV2_1/element_shapeИ
#sequential_1/lstm_3/TensorArrayV2_1TensorListReserve:sequential_1/lstm_3/TensorArrayV2_1/element_shape:output:0,sequential_1/lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_1/lstm_3/TensorArrayV2_1v
sequential_1/lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_1/lstm_3/timeІ
,sequential_1/lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2.
,sequential_1/lstm_3/while/maximum_iterationsТ
&sequential_1/lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_1/lstm_3/while/loop_counterХ
sequential_1/lstm_3/whileWhile/sequential_1/lstm_3/while/loop_counter:output:05sequential_1/lstm_3/while/maximum_iterations:output:0!sequential_1/lstm_3/time:output:0,sequential_1/lstm_3/TensorArrayV2_1:handle:0"sequential_1/lstm_3/zeros:output:0$sequential_1/lstm_3/zeros_1:output:0,sequential_1/lstm_3/strided_slice_1:output:0Ksequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_1_lstm_3_lstm_cell_5_matmul_readvariableop_resource@sequential_1_lstm_3_lstm_cell_5_matmul_1_readvariableop_resource?sequential_1_lstm_3_lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*0
body(R&
$sequential_1_lstm_3_while_body_44541*0
cond(R&
$sequential_1_lstm_3_while_cond_44540*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
sequential_1/lstm_3/whileЁ
Dsequential_1/lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2F
Dsequential_1/lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeє
6sequential_1/lstm_3/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_1/lstm_3/while:output:3Msequential_1/lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:Ц€€€€€€€€€@*
element_dtype028
6sequential_1/lstm_3/TensorArrayV2Stack/TensorListStack©
)sequential_1/lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2+
)sequential_1/lstm_3/strided_slice_3/stack§
+sequential_1/lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_1/lstm_3/strided_slice_3/stack_1§
+sequential_1/lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_3/strided_slice_3/stack_2Т
#sequential_1/lstm_3/strided_slice_3StridedSlice?sequential_1/lstm_3/TensorArrayV2Stack/TensorListStack:tensor:02sequential_1/lstm_3/strided_slice_3/stack:output:04sequential_1/lstm_3/strided_slice_3/stack_1:output:04sequential_1/lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2%
#sequential_1/lstm_3/strided_slice_3°
$sequential_1/lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_1/lstm_3/transpose_1/permц
sequential_1/lstm_3/transpose_1	Transpose?sequential_1/lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_1/lstm_3/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2!
sequential_1/lstm_3/transpose_1О
sequential_1/lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_1/lstm_3/runtimeЃ
sequential_1/dropout_4/IdentityIdentity,sequential_1/lstm_3/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2!
sequential_1/dropout_4/Identityћ
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02,
*sequential_1/dense_2/MatMul/ReadVariableOp‘
sequential_1/dense_2/MatMulMatMul(sequential_1/dropout_4/Identity:output:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_1/dense_2/MatMulЋ
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_1/dense_2/BiasAdd/ReadVariableOp’
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_1/dense_2/BiasAddЧ
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_1/dense_2/Relu©
sequential_1/dropout_5/IdentityIdentity'sequential_1/dense_2/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€2!
sequential_1/dropout_5/Identityћ
*sequential_1/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_1/dense_3/MatMul/ReadVariableOp‘
sequential_1/dense_3/MatMulMatMul(sequential_1/dropout_5/Identity:output:02sequential_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_1/dense_3/MatMulЋ
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_1/dense_3/BiasAdd/ReadVariableOp’
sequential_1/dense_3/BiasAddBiasAdd%sequential_1/dense_3/MatMul:product:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_1/dense_3/BiasAdd†
sequential_1/dense_3/SoftmaxSoftmax%sequential_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_1/dense_3/Softmax≤
IdentityIdentity&sequential_1/dense_3/Softmax:softmax:0^sequential_1/lstm_2/while^sequential_1/lstm_3/while*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:€€€€€€€€€ЦЦ::::::::::26
sequential_1/lstm_2/whilesequential_1/lstm_2/while26
sequential_1/lstm_3/whilesequential_1/lstm_3/while:V R
-
_output_shapes
:€€€€€€€€€ЦЦ
!
_user_specified_name	input_5:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
Ѕф
Ж
G__inference_sequential_1_layer_call_and_return_conditional_losses_47192

inputs5
1lstm_2_lstm_cell_4_matmul_readvariableop_resource7
3lstm_2_lstm_cell_4_matmul_1_readvariableop_resource6
2lstm_2_lstm_cell_4_biasadd_readvariableop_resource5
1lstm_3_lstm_cell_5_matmul_readvariableop_resource7
3lstm_3_lstm_cell_5_matmul_1_readvariableop_resource6
2lstm_3_lstm_cell_5_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identityИҐlstm_2/whileҐlstm_3/whileR
lstm_2/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_2/ShapeВ
lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice/stackЖ
lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_2/strided_slice/stack_1Ж
lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_2/strided_slice/stack_2М
lstm_2/strided_sliceStridedSlicelstm_2/Shape:output:0#lstm_2/strided_slice/stack:output:0%lstm_2/strided_slice/stack_1:output:0%lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_2/strided_slicej
lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_2/zeros/mul/yИ
lstm_2/zeros/mulMullstm_2/strided_slice:output:0lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros/mulm
lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm_2/zeros/Less/yГ
lstm_2/zeros/LessLesslstm_2/zeros/mul:z:0lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros/Lessp
lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_2/zeros/packed/1Я
lstm_2/zeros/packedPacklstm_2/strided_slice:output:0lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_2/zeros/packedm
lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/zeros/ConstС
lstm_2/zerosFilllstm_2/zeros/packed:output:0lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_2/zerosn
lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_2/zeros_1/mul/yО
lstm_2/zeros_1/mulMullstm_2/strided_slice:output:0lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros_1/mulq
lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm_2/zeros_1/Less/yЛ
lstm_2/zeros_1/LessLesslstm_2/zeros_1/mul:z:0lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros_1/Lesst
lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_2/zeros_1/packed/1•
lstm_2/zeros_1/packedPacklstm_2/strided_slice:output:0 lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_2/zeros_1/packedq
lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/zeros_1/ConstЩ
lstm_2/zeros_1Filllstm_2/zeros_1/packed:output:0lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_2/zeros_1Г
lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_2/transpose/permС
lstm_2/transpose	Transposeinputslstm_2/transpose/perm:output:0*
T0*-
_output_shapes
:Ц€€€€€€€€€Ц2
lstm_2/transposed
lstm_2/Shape_1Shapelstm_2/transpose:y:0*
T0*
_output_shapes
:2
lstm_2/Shape_1Ж
lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice_1/stackК
lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_1/stack_1К
lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_1/stack_2Ш
lstm_2/strided_slice_1StridedSlicelstm_2/Shape_1:output:0%lstm_2/strided_slice_1/stack:output:0'lstm_2/strided_slice_1/stack_1:output:0'lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_2/strided_slice_1У
"lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2$
"lstm_2/TensorArrayV2/element_shapeќ
lstm_2/TensorArrayV2TensorListReserve+lstm_2/TensorArrayV2/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_2/TensorArrayV2Ќ
<lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Ц   2>
<lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeФ
.lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_2/transpose:y:0Elstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_2/TensorArrayUnstack/TensorListFromTensorЖ
lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice_2/stackК
lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_2/stack_1К
lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_2/stack_2І
lstm_2/strided_slice_2StridedSlicelstm_2/transpose:y:0%lstm_2/strided_slice_2/stack:output:0'lstm_2/strided_slice_2/stack_1:output:0'lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€Ц*
shrink_axis_mask2
lstm_2/strided_slice_2»
(lstm_2/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp1lstm_2_lstm_cell_4_matmul_readvariableop_resource* 
_output_shapes
:
ЦА*
dtype02*
(lstm_2/lstm_cell_4/MatMul/ReadVariableOp∆
lstm_2/lstm_cell_4/MatMulMatMullstm_2/strided_slice_2:output:00lstm_2/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_2/lstm_cell_4/MatMulЌ
*lstm_2/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp3lstm_2_lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02,
*lstm_2/lstm_cell_4/MatMul_1/ReadVariableOp¬
lstm_2/lstm_cell_4/MatMul_1MatMullstm_2/zeros:output:02lstm_2/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_2/lstm_cell_4/MatMul_1Є
lstm_2/lstm_cell_4/addAddV2#lstm_2/lstm_cell_4/MatMul:product:0%lstm_2/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_2/lstm_cell_4/add∆
)lstm_2/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp2lstm_2_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)lstm_2/lstm_cell_4/BiasAdd/ReadVariableOp≈
lstm_2/lstm_cell_4/BiasAddBiasAddlstm_2/lstm_cell_4/add:z:01lstm_2/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_2/lstm_cell_4/BiasAddv
lstm_2/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_2/lstm_cell_4/ConstК
"lstm_2/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_2/lstm_cell_4/split/split_dimЛ
lstm_2/lstm_cell_4/splitSplit+lstm_2/lstm_cell_4/split/split_dim:output:0#lstm_2/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_2/lstm_cell_4/splitШ
lstm_2/lstm_cell_4/SigmoidSigmoid!lstm_2/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_2/lstm_cell_4/SigmoidЬ
lstm_2/lstm_cell_4/Sigmoid_1Sigmoid!lstm_2/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_2/lstm_cell_4/Sigmoid_1§
lstm_2/lstm_cell_4/mulMul lstm_2/lstm_cell_4/Sigmoid_1:y:0lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_2/lstm_cell_4/mulП
lstm_2/lstm_cell_4/ReluRelu!lstm_2/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_2/lstm_cell_4/Reluі
lstm_2/lstm_cell_4/mul_1Mullstm_2/lstm_cell_4/Sigmoid:y:0%lstm_2/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_2/lstm_cell_4/mul_1©
lstm_2/lstm_cell_4/add_1AddV2lstm_2/lstm_cell_4/mul:z:0lstm_2/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_2/lstm_cell_4/add_1Ь
lstm_2/lstm_cell_4/Sigmoid_2Sigmoid!lstm_2/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_2/lstm_cell_4/Sigmoid_2О
lstm_2/lstm_cell_4/Relu_1Relulstm_2/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_2/lstm_cell_4/Relu_1Є
lstm_2/lstm_cell_4/mul_2Mul lstm_2/lstm_cell_4/Sigmoid_2:y:0'lstm_2/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_2/lstm_cell_4/mul_2Э
$lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2&
$lstm_2/TensorArrayV2_1/element_shape‘
lstm_2/TensorArrayV2_1TensorListReserve-lstm_2/TensorArrayV2_1/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_2/TensorArrayV2_1\
lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_2/timeН
lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2!
lstm_2/while/maximum_iterationsx
lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_2/while/loop_counter“
lstm_2/whileWhile"lstm_2/while/loop_counter:output:0(lstm_2/while/maximum_iterations:output:0lstm_2/time:output:0lstm_2/TensorArrayV2_1:handle:0lstm_2/zeros:output:0lstm_2/zeros_1:output:0lstm_2/strided_slice_1:output:0>lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_2_lstm_cell_4_matmul_readvariableop_resource3lstm_2_lstm_cell_4_matmul_1_readvariableop_resource2lstm_2_lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*#
bodyR
lstm_2_while_body_46920*#
condR
lstm_2_while_cond_46919*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
lstm_2/while√
7lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   29
7lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeЕ
)lstm_2/TensorArrayV2Stack/TensorListStackTensorListStacklstm_2/while:output:3@lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:Ц€€€€€€€€€@*
element_dtype02+
)lstm_2/TensorArrayV2Stack/TensorListStackП
lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
lstm_2/strided_slice_3/stackК
lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_2/strided_slice_3/stack_1К
lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_3/stack_2ƒ
lstm_2/strided_slice_3StridedSlice2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_2/strided_slice_3/stack:output:0'lstm_2/strided_slice_3/stack_1:output:0'lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
lstm_2/strided_slice_3З
lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_2/transpose_1/perm¬
lstm_2/transpose_1	Transpose2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_2/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2
lstm_2/transpose_1t
lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/runtimew
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout_3/dropout/Const¶
dropout_3/dropout/MulMullstm_2/transpose_1:y:0 dropout_3/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2
dropout_3/dropout/Mulx
dropout_3/dropout/ShapeShapelstm_2/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape„
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@*
dtype020
.dropout_3/dropout/random_uniform/RandomUniformЙ
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2"
 dropout_3/dropout/GreaterEqual/yл
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2 
dropout_3/dropout/GreaterEqualҐ
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€Ц@2
dropout_3/dropout/CastІ
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2
dropout_3/dropout/Mul_1g
lstm_3/ShapeShapedropout_3/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_3/ShapeВ
lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice/stackЖ
lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_1Ж
lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_2М
lstm_3/strided_sliceStridedSlicelstm_3/Shape:output:0#lstm_3/strided_slice/stack:output:0%lstm_3/strided_slice/stack_1:output:0%lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slicej
lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_3/zeros/mul/yИ
lstm_3/zeros/mulMullstm_3/strided_slice:output:0lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/mulm
lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm_3/zeros/Less/yГ
lstm_3/zeros/LessLesslstm_3/zeros/mul:z:0lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/Lessp
lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_3/zeros/packed/1Я
lstm_3/zeros/packedPacklstm_3/strided_slice:output:0lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros/packedm
lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros/ConstС
lstm_3/zerosFilllstm_3/zeros/packed:output:0lstm_3/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_3/zerosn
lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_3/zeros_1/mul/yО
lstm_3/zeros_1/mulMullstm_3/strided_slice:output:0lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/mulq
lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm_3/zeros_1/Less/yЛ
lstm_3/zeros_1/LessLesslstm_3/zeros_1/mul:z:0lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/Lesst
lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_3/zeros_1/packed/1•
lstm_3/zeros_1/packedPacklstm_3/strided_slice:output:0 lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros_1/packedq
lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros_1/ConstЩ
lstm_3/zeros_1Filllstm_3/zeros_1/packed:output:0lstm_3/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_3/zeros_1Г
lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_3/transpose/perm•
lstm_3/transpose	Transposedropout_3/dropout/Mul_1:z:0lstm_3/transpose/perm:output:0*
T0*,
_output_shapes
:Ц€€€€€€€€€@2
lstm_3/transposed
lstm_3/Shape_1Shapelstm_3/transpose:y:0*
T0*
_output_shapes
:2
lstm_3/Shape_1Ж
lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice_1/stackК
lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_1/stack_1К
lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_1/stack_2Ш
lstm_3/strided_slice_1StridedSlicelstm_3/Shape_1:output:0%lstm_3/strided_slice_1/stack:output:0'lstm_3/strided_slice_1/stack_1:output:0'lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slice_1У
"lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2$
"lstm_3/TensorArrayV2/element_shapeќ
lstm_3/TensorArrayV2TensorListReserve+lstm_3/TensorArrayV2/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_3/TensorArrayV2Ќ
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2>
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeФ
.lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_3/transpose:y:0Elstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_3/TensorArrayUnstack/TensorListFromTensorЖ
lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice_2/stackК
lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_2/stack_1К
lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_2/stack_2¶
lstm_3/strided_slice_2StridedSlicelstm_3/transpose:y:0%lstm_3/strided_slice_2/stack:output:0'lstm_3/strided_slice_2/stack_1:output:0'lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
lstm_3/strided_slice_2«
(lstm_3/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp1lstm_3_lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02*
(lstm_3/lstm_cell_5/MatMul/ReadVariableOp∆
lstm_3/lstm_cell_5/MatMulMatMullstm_3/strided_slice_2:output:00lstm_3/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_3/lstm_cell_5/MatMulЌ
*lstm_3/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp3lstm_3_lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02,
*lstm_3/lstm_cell_5/MatMul_1/ReadVariableOp¬
lstm_3/lstm_cell_5/MatMul_1MatMullstm_3/zeros:output:02lstm_3/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_3/lstm_cell_5/MatMul_1Є
lstm_3/lstm_cell_5/addAddV2#lstm_3/lstm_cell_5/MatMul:product:0%lstm_3/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_3/lstm_cell_5/add∆
)lstm_3/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp2lstm_3_lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)lstm_3/lstm_cell_5/BiasAdd/ReadVariableOp≈
lstm_3/lstm_cell_5/BiasAddBiasAddlstm_3/lstm_cell_5/add:z:01lstm_3/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_3/lstm_cell_5/BiasAddv
lstm_3/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/lstm_cell_5/ConstК
"lstm_3/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_3/lstm_cell_5/split/split_dimЛ
lstm_3/lstm_cell_5/splitSplit+lstm_3/lstm_cell_5/split/split_dim:output:0#lstm_3/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_3/lstm_cell_5/splitШ
lstm_3/lstm_cell_5/SigmoidSigmoid!lstm_3/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_3/lstm_cell_5/SigmoidЬ
lstm_3/lstm_cell_5/Sigmoid_1Sigmoid!lstm_3/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_3/lstm_cell_5/Sigmoid_1§
lstm_3/lstm_cell_5/mulMul lstm_3/lstm_cell_5/Sigmoid_1:y:0lstm_3/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_3/lstm_cell_5/mulП
lstm_3/lstm_cell_5/ReluRelu!lstm_3/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_3/lstm_cell_5/Reluі
lstm_3/lstm_cell_5/mul_1Mullstm_3/lstm_cell_5/Sigmoid:y:0%lstm_3/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_3/lstm_cell_5/mul_1©
lstm_3/lstm_cell_5/add_1AddV2lstm_3/lstm_cell_5/mul:z:0lstm_3/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_3/lstm_cell_5/add_1Ь
lstm_3/lstm_cell_5/Sigmoid_2Sigmoid!lstm_3/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_3/lstm_cell_5/Sigmoid_2О
lstm_3/lstm_cell_5/Relu_1Relulstm_3/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_3/lstm_cell_5/Relu_1Є
lstm_3/lstm_cell_5/mul_2Mul lstm_3/lstm_cell_5/Sigmoid_2:y:0'lstm_3/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_3/lstm_cell_5/mul_2Э
$lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2&
$lstm_3/TensorArrayV2_1/element_shape‘
lstm_3/TensorArrayV2_1TensorListReserve-lstm_3/TensorArrayV2_1/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_3/TensorArrayV2_1\
lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/timeН
lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2!
lstm_3/while/maximum_iterationsx
lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/while/loop_counter“
lstm_3/whileWhile"lstm_3/while/loop_counter:output:0(lstm_3/while/maximum_iterations:output:0lstm_3/time:output:0lstm_3/TensorArrayV2_1:handle:0lstm_3/zeros:output:0lstm_3/zeros_1:output:0lstm_3/strided_slice_1:output:0>lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_3_lstm_cell_5_matmul_readvariableop_resource3lstm_3_lstm_cell_5_matmul_1_readvariableop_resource2lstm_3_lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*#
bodyR
lstm_3_while_body_47077*#
condR
lstm_3_while_cond_47076*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
lstm_3/while√
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   29
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeЕ
)lstm_3/TensorArrayV2Stack/TensorListStackTensorListStacklstm_3/while:output:3@lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:Ц€€€€€€€€€@*
element_dtype02+
)lstm_3/TensorArrayV2Stack/TensorListStackП
lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
lstm_3/strided_slice_3/stackК
lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_3/strided_slice_3/stack_1К
lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_3/stack_2ƒ
lstm_3/strided_slice_3StridedSlice2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_3/strided_slice_3/stack:output:0'lstm_3/strided_slice_3/stack_1:output:0'lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
lstm_3/strided_slice_3З
lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_3/transpose_1/perm¬
lstm_3/transpose_1	Transpose2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_3/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2
lstm_3/transpose_1t
lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/runtimew
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout_4/dropout/Const™
dropout_4/dropout/MulMullstm_3/strided_slice_3:output:0 dropout_4/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_4/dropout/MulБ
dropout_4/dropout/ShapeShapelstm_3/strided_slice_3:output:0*
T0*
_output_shapes
:2
dropout_4/dropout/Shape“
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype020
.dropout_4/dropout/random_uniform/RandomUniformЙ
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2"
 dropout_4/dropout/GreaterEqual/yж
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2 
dropout_4/dropout/GreaterEqualЭ
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout_4/dropout/CastҐ
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_4/dropout/Mul_1•
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_2/MatMul/ReadVariableOp†
dense_2/MatMulMatMuldropout_4/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_2/MatMul§
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp°
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_2/Reluw
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout_5/dropout/Const•
dropout_5/dropout/MulMuldense_2/Relu:activations:0 dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dropout_5/dropout/Mul|
dropout_5/dropout/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:2
dropout_5/dropout/Shape“
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype020
.dropout_5/dropout/random_uniform/RandomUniformЙ
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2"
 dropout_5/dropout/GreaterEqual/yж
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2 
dropout_5/dropout/GreaterEqualЭ
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€2
dropout_5/dropout/CastҐ
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€2
dropout_5/dropout/Mul_1•
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp†
dense_3/MatMulMatMuldropout_5/dropout/Mul_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_3/MatMul§
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp°
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_3/BiasAddy
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_3/SoftmaxЛ
IdentityIdentitydense_3/Softmax:softmax:0^lstm_2/while^lstm_3/while*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:€€€€€€€€€ЦЦ::::::::::2
lstm_2/whilelstm_2/while2
lstm_3/whilelstm_3/while:U Q
-
_output_shapes
:€€€€€€€€€ЦЦ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
в2
п
while_body_48312
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
,lstm_cell_5_matmul_readvariableop_resource_02
.lstm_cell_5_matmul_1_readvariableop_resource_01
-lstm_cell_5_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resourceИЈ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   23
1TensorArrayV2Read/TensorListGetItem/element_shapeµ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€@*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemі
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOpЉ
lstm_cell_5/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMulЇ
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp•
lstm_cell_5/MatMul_1MatMulplaceholder_2+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMul_1Ь
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/add≥
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp©
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dimп
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_5/splitГ
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/SigmoidЗ
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_1Е
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/ReluШ
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_1Н
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/add_1З
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Relu_1Ь
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_2Ѕ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3m

Identity_4Identitylstm_cell_5/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_4m

Identity_5Identitylstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"\
+lstm_cell_5_biasadd_readvariableop_resource-lstm_cell_5_biasadd_readvariableop_resource_0"^
,lstm_cell_5_matmul_1_readvariableop_resource.lstm_cell_5_matmul_1_readvariableop_resource_0"Z
*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :€€€€€€€€€@:€€€€€€€€€@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
«
Ѓ
lstm_2_while_cond_46919
lstm_2_while_loop_counter#
lstm_2_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_2_strided_slice_14
0lstm_2_while_cond_46919___redundant_placeholder04
0lstm_2_while_cond_46919___redundant_placeholder14
0lstm_2_while_cond_46919___redundant_placeholder24
0lstm_2_while_cond_46919___redundant_placeholder3
identity
_
LessLessplaceholderless_lstm_2_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Ѕ
О
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_44715

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ЦА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimњ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:€€€€€€€€€Ц:€€€€€€€€€@:€€€€€€€€€@::::P L
(
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_namestates:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
«
Ѓ
lstm_2_while_cond_47259
lstm_2_while_loop_counter#
lstm_2_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_2_strided_slice_14
0lstm_2_while_cond_47259___redundant_placeholder04
0lstm_2_while_cond_47259___redundant_placeholder14
0lstm_2_while_cond_47259___redundant_placeholder24
0lstm_2_while_cond_47259___redundant_placeholder3
identity
_
LessLessplaceholderless_lstm_2_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Д
E
)__inference_dropout_3_layer_call_fn_48244

inputs
identity•
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_output_shapes
:€€€€€€€€€Ц@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_462182
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2

Identity"
identityIdentity:output:0*+
_input_shapes
:€€€€€€€€€Ц@:T P
,
_output_shapes
:€€€€€€€€€Ц@
 
_user_specified_nameinputs
д2
п
while_body_47782
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
,lstm_cell_4_matmul_readvariableop_resource_02
.lstm_cell_4_matmul_1_readvariableop_resource_01
-lstm_cell_4_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resourceИЈ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Ц   23
1TensorArrayV2Read/TensorListGetItem/element_shapeґ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€Ц*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemµ
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_readvariableop_resource_0* 
_output_shapes
:
ЦА*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOpЉ
lstm_cell_4/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMulЇ
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp•
lstm_cell_4/MatMul_1MatMulplaceholder_2+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMul_1Ь
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/add≥
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp©
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimп
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_4/splitГ
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/SigmoidЗ
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_1Е
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/ReluШ
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_1Н
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_1З
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_2Ѕ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3m

Identity_4Identitylstm_cell_4/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_4m

Identity_5Identitylstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"\
+lstm_cell_4_biasadd_readvariableop_resource-lstm_cell_4_biasadd_readvariableop_resource_0"^
,lstm_cell_4_matmul_1_readvariableop_resource.lstm_cell_4_matmul_1_readvariableop_resource_0"Z
*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :€€€€€€€€€@:€€€€€€€€€@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
«
Ѓ
lstm_3_while_cond_47409
lstm_3_while_loop_counter#
lstm_3_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_3_strided_slice_14
0lstm_3_while_cond_47409___redundant_placeholder04
0lstm_3_while_cond_47409___redundant_placeholder14
0lstm_3_while_cond_47409___redundant_placeholder24
0lstm_3_while_cond_47409___redundant_placeholder3
identity
_
LessLessplaceholderless_lstm_3_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
т
|
'__inference_dense_3_layer_call_fn_48994

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCall–
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_466642
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
СX
э
A__inference_lstm_3_layer_call_and_return_conditional_losses_48397
inputs_0.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resource
identityИҐwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЕ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_2≤
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOp™
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMulЄ
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp¶
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMul_1Ь
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/add±
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp©
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dimп
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_5/splitГ
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/SigmoidЗ
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_1И
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/ReluШ
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_1Н
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/add_1З
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Relu_1Ь
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
TensorArrayV2_1/element_shapeЄ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterй
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_48312*
condR
while_cond_48311*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€€€€€€€€€€@:::2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
"
_user_specified_name
inputs/0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
И
ц
while_cond_46085
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_46085___redundant_placeholder0-
)while_cond_46085___redundant_placeholder1-
)while_cond_46085___redundant_placeholder2-
)while_cond_46085___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
бW
ы
A__inference_lstm_2_layer_call_and_return_conditional_losses_48042

inputs.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resource
identityИҐwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm|
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:Ц€€€€€€€€€Ц2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Ц   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2э
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€Ц*
shrink_axis_mask2
strided_slice_2≥
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource* 
_output_shapes
:
ЦА*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOp™
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMulЄ
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp¶
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMul_1Ь
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/add±
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp©
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimп
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_4/splitГ
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/SigmoidЗ
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_1И
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/ReluШ
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_1Н
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_1З
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
TensorArrayV2_1/element_shapeЄ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterй
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_47957*
condR
while_cond_47956*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   22
0TensorArrayV2Stack/TensorListStack/element_shapeй
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:Ц€€€€€€€€€@*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¶
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimep
IdentityIdentitytranspose_1:y:0^while*
T0*,
_output_shapes
:€€€€€€€€€Ц@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€ЦЦ:::2
whilewhile:U Q
-
_output_shapes
:€€€€€€€€€ЦЦ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
©
c
D__inference_dropout_3_layer_call_and_return_conditional_losses_48229

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeє
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/y√
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€Ц@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2

Identity"
identityIdentity:output:0*+
_input_shapes
:€€€€€€€€€Ц@:T P
,
_output_shapes
:€€€€€€€€€Ц@
 
_user_specified_nameinputs
в2
п
while_body_48640
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
,lstm_cell_5_matmul_readvariableop_resource_02
.lstm_cell_5_matmul_1_readvariableop_resource_01
-lstm_cell_5_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resourceИЈ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   23
1TensorArrayV2Read/TensorListGetItem/element_shapeµ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€@*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemі
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOpЉ
lstm_cell_5/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMulЇ
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp•
lstm_cell_5/MatMul_1MatMulplaceholder_2+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMul_1Ь
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/add≥
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp©
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dimп
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_5/splitГ
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/SigmoidЗ
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_1Е
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/ReluШ
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_1Н
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/add_1З
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Relu_1Ь
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_2Ѕ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3m

Identity_4Identitylstm_cell_5/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_4m

Identity_5Identitylstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"\
+lstm_cell_5_biasadd_readvariableop_resource-lstm_cell_5_biasadd_readvariableop_resource_0"^
,lstm_cell_5_matmul_1_readvariableop_resource.lstm_cell_5_matmul_1_readvariableop_resource_0"Z
*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :€€€€€€€€€@:€€€€€€€€€@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
«
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_48917

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Љ	
Ц
$sequential_1_lstm_3_while_cond_44540*
&sequential_1_lstm_3_while_loop_counter0
,sequential_1_lstm_3_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3,
(less_sequential_1_lstm_3_strided_slice_1A
=sequential_1_lstm_3_while_cond_44540___redundant_placeholder0A
=sequential_1_lstm_3_while_cond_44540___redundant_placeholder1A
=sequential_1_lstm_3_while_cond_44540___redundant_placeholder2A
=sequential_1_lstm_3_while_cond_44540___redundant_placeholder3
identity
l
LessLessplaceholder(less_sequential_1_lstm_3_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
д2
п
while_body_47629
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
,lstm_cell_4_matmul_readvariableop_resource_02
.lstm_cell_4_matmul_1_readvariableop_resource_01
-lstm_cell_4_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resourceИЈ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Ц   23
1TensorArrayV2Read/TensorListGetItem/element_shapeґ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€Ц*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemµ
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_readvariableop_resource_0* 
_output_shapes
:
ЦА*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOpЉ
lstm_cell_4/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMulЇ
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp•
lstm_cell_4/MatMul_1MatMulplaceholder_2+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMul_1Ь
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/add≥
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp©
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimп
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_4/splitГ
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/SigmoidЗ
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_1Е
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/ReluШ
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_1Н
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_1З
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_2Ѕ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3m

Identity_4Identitylstm_cell_4/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_4m

Identity_5Identitylstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"\
+lstm_cell_4_biasadd_readvariableop_resource-lstm_cell_4_biasadd_readvariableop_resource_0"^
,lstm_cell_4_matmul_1_readvariableop_resource.lstm_cell_4_matmul_1_readvariableop_resource_0"Z
*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :€€€€€€€€€@:€€€€€€€€€@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
бW
ы
A__inference_lstm_2_layer_call_and_return_conditional_losses_46171

inputs.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resource
identityИҐwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm|
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:Ц€€€€€€€€€Ц2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Ц   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2э
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€Ц*
shrink_axis_mask2
strided_slice_2≥
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource* 
_output_shapes
:
ЦА*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOp™
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMulЄ
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp¶
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMul_1Ь
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/add±
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp©
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimп
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_4/splitГ
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/SigmoidЗ
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_1И
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/ReluШ
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_1Н
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_1З
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
TensorArrayV2_1/element_shapeЄ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterй
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_46086*
condR
while_cond_46085*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   22
0TensorArrayV2Stack/TensorListStack/element_shapeй
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:Ц€€€€€€€€€@*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¶
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimep
IdentityIdentitytranspose_1:y:0^while*
T0*,
_output_shapes
:€€€€€€€€€Ц@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€ЦЦ:::2
whilewhile:U Q
-
_output_shapes
:€€€€€€€€€ЦЦ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
џ
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_46218

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:€€€€€€€€€Ц@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:€€€€€€€€€Ц@:T P
,
_output_shapes
:€€€€€€€€€Ц@
 
_user_specified_nameinputs
ь

ъ
,__inference_sequential_1_layer_call_fn_46825
input_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identityИҐStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*,
_read_only_resource_inputs

	
**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_468022
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:€€€€€€€€€ЦЦ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
-
_output_shapes
:€€€€€€€€€ЦЦ
!
_user_specified_name	input_5:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
г
™
B__inference_dense_2_layer_call_and_return_conditional_losses_48938

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@:::O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Є 
щ
while_body_45652
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
lstm_cell_5_45676_0
lstm_cell_5_45678_0
lstm_cell_5_45680_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
lstm_cell_5_45676
lstm_cell_5_45678
lstm_cell_5_45680ИҐ#lstm_cell_5/StatefulPartitionedCallЈ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   23
1TensorArrayV2Read/TensorListGetItem/element_shapeµ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€@*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemЕ
#lstm_cell_5/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3lstm_cell_5_45676_0lstm_cell_5_45678_0lstm_cell_5_45680_0*
Tin

2*
Tout
2*M
_output_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_453252%
#lstm_cell_5/StatefulPartitionedCallЎ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder,lstm_cell_5/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1r
IdentityIdentity	add_1:z:0$^lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2

IdentityЕ

Identity_1Identitywhile_maximum_iterations$^lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1t

Identity_2Identityadd:z:0$^lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2°

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0$^lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3™

Identity_4Identity,lstm_cell_5/StatefulPartitionedCall:output:1$^lstm_cell_5/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_4™

Identity_5Identity,lstm_cell_5/StatefulPartitionedCall:output:2$^lstm_cell_5/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"(
lstm_cell_5_45676lstm_cell_5_45676_0"(
lstm_cell_5_45678lstm_cell_5_45678_0"(
lstm_cell_5_45680lstm_cell_5_45680_0"$
strided_slice_1strided_slice_1_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :€€€€€€€€€@:€€€€€€€€€@: : :::2J
#lstm_cell_5/StatefulPartitionedCall#lstm_cell_5/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
т
|
'__inference_dense_2_layer_call_fn_48947

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCall–
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_466072
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
є
К
&__inference_lstm_3_layer_call_fn_48900

inputs
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_465362
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€Ц@:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€Ц@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ь4
ы
$sequential_1_lstm_2_while_body_44391*
&sequential_1_lstm_2_while_loop_counter0
,sequential_1_lstm_2_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3)
%sequential_1_lstm_2_strided_slice_1_0e
atensorarrayv2read_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_tensorlistfromtensor_00
,lstm_cell_4_matmul_readvariableop_resource_02
.lstm_cell_4_matmul_1_readvariableop_resource_01
-lstm_cell_4_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5'
#sequential_1_lstm_2_strided_slice_1c
_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_tensorlistfromtensor.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resourceИЈ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Ц   23
1TensorArrayV2Read/TensorListGetItem/element_shape 
#TensorArrayV2Read/TensorListGetItemTensorListGetItematensorarrayv2read_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€Ц*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemµ
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_readvariableop_resource_0* 
_output_shapes
:
ЦА*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOpЉ
lstm_cell_4/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMulЇ
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp•
lstm_cell_4/MatMul_1MatMulplaceholder_2+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMul_1Ь
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/add≥
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp©
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimп
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_4/splitГ
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/SigmoidЗ
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_1Е
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/ReluШ
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_1Н
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_1З
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_2Ѕ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yr
add_1AddV2&sequential_1_lstm_2_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identitys

Identity_1Identity,sequential_1_lstm_2_while_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3m

Identity_4Identitylstm_cell_4/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_4m

Identity_5Identitylstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"\
+lstm_cell_4_biasadd_readvariableop_resource-lstm_cell_4_biasadd_readvariableop_resource_0"^
,lstm_cell_4_matmul_1_readvariableop_resource.lstm_cell_4_matmul_1_readvariableop_resource_0"Z
*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_readvariableop_resource_0"L
#sequential_1_lstm_2_strided_slice_1%sequential_1_lstm_2_strided_slice_1_0"ƒ
_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_tensorlistfromtensoratensorarrayv2read_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :€€€€€€€€€@:€€€€€€€€€@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
аW
ы
A__inference_lstm_3_layer_call_and_return_conditional_losses_48725

inputs.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resource
identityИҐwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:Ц€€€€€€€€€@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_2≤
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOp™
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMulЄ
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp¶
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMul_1Ь
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/add±
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp©
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dimп
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_5/splitГ
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/SigmoidЗ
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_1И
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/ReluШ
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_1Н
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/add_1З
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Relu_1Ь
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
TensorArrayV2_1/element_shapeЄ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterй
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_48640*
condR
while_cond_48639*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   22
0TensorArrayV2Stack/TensorListStack/element_shapeй
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:Ц€€€€€€€€€@*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¶
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€Ц@:::2
whilewhile:T P
,
_output_shapes
:€€€€€€€€€Ц@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
эD
”
A__inference_lstm_2_layer_call_and_return_conditional_losses_45243

inputs
lstm_cell_4_45161
lstm_cell_4_45163
lstm_cell_4_45165
identityИҐ#lstm_cell_4/StatefulPartitionedCallҐwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permД
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ц2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Ц   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2э
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€Ц*
shrink_axis_mask2
strided_slice_2с
#lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_4_45161lstm_cell_4_45163lstm_cell_4_45165*
Tin

2*
Tout
2*M
_output_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_447482%
#lstm_cell_4/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
TensorArrayV2_1/element_shapeЄ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЫ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_4_45161lstm_cell_4_45163lstm_cell_4_45165*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_45174*
condR
while_cond_45173*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeЮ
IdentityIdentitytranspose_1:y:0$^lstm_cell_4/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:€€€€€€€€€€€€€€€€€€Ц:::2J
#lstm_cell_4/StatefulPartitionedCall#lstm_cell_4/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ц
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
є
К
&__inference_lstm_3_layer_call_fn_48889

inputs
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_463832
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€Ц@:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€Ц@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
И
ц
while_cond_47956
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_47956___redundant_placeholder0-
)while_cond_47956___redundant_placeholder1-
)while_cond_47956___redundant_placeholder2-
)while_cond_47956___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
И
ц
while_cond_47628
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_47628___redundant_placeholder0-
)while_cond_47628___redundant_placeholder1-
)while_cond_47628___redundant_placeholder2-
)while_cond_47628___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
ѕ
М
&__inference_lstm_3_layer_call_fn_48561
inputs_0
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_457212
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€€€€€€€€€€@:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
"
_user_specified_name
inputs/0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
≈
К
&__inference_lstm_2_layer_call_fn_48217

inputs
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*,
_output_shapes
:€€€€€€€€€Ц@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_461712
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€Ц@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€ЦЦ:::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:€€€€€€€€€ЦЦ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
л
М
&__inference_lstm_2_layer_call_fn_47889
inputs_0
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_452432
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:€€€€€€€€€€€€€€€€€€Ц:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ц
"
_user_specified_name
inputs/0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
И
ц
while_cond_45932
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_45932___redundant_placeholder0-
)while_cond_45932___redundant_placeholder1-
)while_cond_45932___redundant_placeholder2-
)while_cond_45932___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
И
ц
while_cond_45783
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_45783___redundant_placeholder0-
)while_cond_45783___redundant_placeholder1-
)while_cond_45783___redundant_placeholder2-
)while_cond_45783___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
є 
щ
while_body_45042
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
lstm_cell_4_45066_0
lstm_cell_4_45068_0
lstm_cell_4_45070_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
lstm_cell_4_45066
lstm_cell_4_45068
lstm_cell_4_45070ИҐ#lstm_cell_4/StatefulPartitionedCallЈ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Ц   23
1TensorArrayV2Read/TensorListGetItem/element_shapeґ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€Ц*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemЕ
#lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3lstm_cell_4_45066_0lstm_cell_4_45068_0lstm_cell_4_45070_0*
Tin

2*
Tout
2*M
_output_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_447152%
#lstm_cell_4/StatefulPartitionedCallЎ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder,lstm_cell_4/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1r
IdentityIdentity	add_1:z:0$^lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

IdentityЕ

Identity_1Identitywhile_maximum_iterations$^lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1t

Identity_2Identityadd:z:0$^lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2°

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0$^lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3™

Identity_4Identity,lstm_cell_4/StatefulPartitionedCall:output:1$^lstm_cell_4/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_4™

Identity_5Identity,lstm_cell_4/StatefulPartitionedCall:output:2$^lstm_cell_4/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"(
lstm_cell_4_45066lstm_cell_4_45066_0"(
lstm_cell_4_45068lstm_cell_4_45068_0"(
lstm_cell_4_45070lstm_cell_4_45070_0"$
strided_slice_1strided_slice_1_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :€€€€€€€€€@:€€€€€€€€€@: : :::2J
#lstm_cell_4/StatefulPartitionedCall#lstm_cell_4/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
И
ц
while_cond_46450
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_46450___redundant_placeholder0-
)while_cond_46450___redundant_placeholder1-
)while_cond_46450___redundant_placeholder2-
)while_cond_46450___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
р
E
)__inference_dropout_5_layer_call_fn_48974

inputs
identity†
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_466402
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
И
ц
while_cond_48311
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_48311___redundant_placeholder0-
)while_cond_48311___redundant_placeholder1-
)while_cond_48311___redundant_placeholder2-
)while_cond_48311___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Є 
щ
while_body_45784
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
lstm_cell_5_45808_0
lstm_cell_5_45810_0
lstm_cell_5_45812_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
lstm_cell_5_45808
lstm_cell_5_45810
lstm_cell_5_45812ИҐ#lstm_cell_5/StatefulPartitionedCallЈ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   23
1TensorArrayV2Read/TensorListGetItem/element_shapeµ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€@*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemЕ
#lstm_cell_5/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3lstm_cell_5_45808_0lstm_cell_5_45810_0lstm_cell_5_45812_0*
Tin

2*
Tout
2*M
_output_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_453582%
#lstm_cell_5/StatefulPartitionedCallЎ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder,lstm_cell_5/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1r
IdentityIdentity	add_1:z:0$^lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2

IdentityЕ

Identity_1Identitywhile_maximum_iterations$^lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1t

Identity_2Identityadd:z:0$^lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2°

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0$^lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3™

Identity_4Identity,lstm_cell_5/StatefulPartitionedCall:output:1$^lstm_cell_5/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_4™

Identity_5Identity,lstm_cell_5/StatefulPartitionedCall:output:2$^lstm_cell_5/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"(
lstm_cell_5_45808lstm_cell_5_45808_0"(
lstm_cell_5_45810lstm_cell_5_45810_0"(
lstm_cell_5_45812lstm_cell_5_45812_0"$
strided_slice_1strided_slice_1_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :€€€€€€€€€@:€€€€€€€€€@: : :::2J
#lstm_cell_5/StatefulPartitionedCall#lstm_cell_5/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
ъ4
ы
$sequential_1_lstm_3_while_body_44541*
&sequential_1_lstm_3_while_loop_counter0
,sequential_1_lstm_3_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3)
%sequential_1_lstm_3_strided_slice_1_0e
atensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensor_00
,lstm_cell_5_matmul_readvariableop_resource_02
.lstm_cell_5_matmul_1_readvariableop_resource_01
-lstm_cell_5_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5'
#sequential_1_lstm_3_strided_slice_1c
_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensor.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resourceИЈ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   23
1TensorArrayV2Read/TensorListGetItem/element_shape…
#TensorArrayV2Read/TensorListGetItemTensorListGetItematensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€@*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemі
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOpЉ
lstm_cell_5/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMulЇ
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp•
lstm_cell_5/MatMul_1MatMulplaceholder_2+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMul_1Ь
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/add≥
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp©
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dimп
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_5/splitГ
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/SigmoidЗ
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_1Е
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/ReluШ
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_1Н
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/add_1З
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Relu_1Ь
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_2Ѕ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yr
add_1AddV2&sequential_1_lstm_3_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identitys

Identity_1Identity,sequential_1_lstm_3_while_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3m

Identity_4Identitylstm_cell_5/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_4m

Identity_5Identitylstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"\
+lstm_cell_5_biasadd_readvariableop_resource-lstm_cell_5_biasadd_readvariableop_resource_0"^
,lstm_cell_5_matmul_1_readvariableop_resource.lstm_cell_5_matmul_1_readvariableop_resource_0"Z
*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_readvariableop_resource_0"L
#sequential_1_lstm_3_strided_slice_1%sequential_1_lstm_3_strided_slice_1_0"ƒ
_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensoratensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :€€€€€€€€€@:€€€€€€€€€@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
∆
Р
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_49160

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimњ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@::::O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
п
Ћ
+__inference_lstm_cell_5_layer_call_fn_49194

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2ИҐStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*M
_output_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_453582
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1Т

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ь
b
)__inference_dropout_4_layer_call_fn_48922

inputs
identityИҐStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_465782
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ѕ
М
&__inference_lstm_3_layer_call_fn_48572
inputs_0
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_458532
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€€€€€€€€€€@:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
"
_user_specified_name
inputs/0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
с
Ћ
+__inference_lstm_cell_4_layer_call_fn_49077

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2ИҐStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*M
_output_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_447152
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1Т

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:€€€€€€€€€Ц:€€€€€€€€€@:€€€€€€€€€@:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
∆3
†
lstm_2_while_body_47260
lstm_2_while_loop_counter#
lstm_2_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_2_strided_slice_1_0X
Ttensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_00
,lstm_cell_4_matmul_readvariableop_resource_02
.lstm_cell_4_matmul_1_readvariableop_resource_01
-lstm_cell_4_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_2_strided_slice_1V
Rtensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resourceИЈ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Ц   23
1TensorArrayV2Read/TensorListGetItem/element_shapeљ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemTtensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€Ц*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemµ
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_readvariableop_resource_0* 
_output_shapes
:
ЦА*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOpЉ
lstm_cell_4/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMulЇ
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp•
lstm_cell_4/MatMul_1MatMulplaceholder_2+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMul_1Ь
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/add≥
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp©
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimп
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_4/splitГ
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/SigmoidЗ
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_1Е
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/ReluШ
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_1Н
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_1З
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_2Ѕ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/ye
add_1AddV2lstm_2_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identityf

Identity_1Identitylstm_2_while_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3m

Identity_4Identitylstm_cell_4/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_4m

Identity_5Identitylstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"2
lstm_2_strided_slice_1lstm_2_strided_slice_1_0"\
+lstm_cell_4_biasadd_readvariableop_resource-lstm_cell_4_biasadd_readvariableop_resource_0"^
,lstm_cell_4_matmul_1_readvariableop_resource.lstm_cell_4_matmul_1_readvariableop_resource_0"Z
*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_readvariableop_resource_0"™
Rtensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensorTtensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :€€€€€€€€€@:€€€€€€€€€@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
џ
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_48234

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:€€€€€€€€€Ц@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:€€€€€€€€€Ц@:T P
,
_output_shapes
:€€€€€€€€€Ц@
 
_user_specified_nameinputs
Њ
О
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_45358

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimњ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@::::O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_namestates:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
в2
п
while_body_46451
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
,lstm_cell_5_matmul_readvariableop_resource_02
.lstm_cell_5_matmul_1_readvariableop_resource_01
-lstm_cell_5_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resourceИЈ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   23
1TensorArrayV2Read/TensorListGetItem/element_shapeµ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€@*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemі
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOpЉ
lstm_cell_5/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMulЇ
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp•
lstm_cell_5/MatMul_1MatMulplaceholder_2+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMul_1Ь
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/add≥
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp©
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dimп
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_5/splitГ
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/SigmoidЗ
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_1Е
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/ReluШ
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_1Н
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/add_1З
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Relu_1Ь
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_2Ѕ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3m

Identity_4Identitylstm_cell_5/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_4m

Identity_5Identitylstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"\
+lstm_cell_5_biasadd_readvariableop_resource-lstm_cell_5_biasadd_readvariableop_resource_0"^
,lstm_cell_5_matmul_1_readvariableop_resource.lstm_cell_5_matmul_1_readvariableop_resource_0"Z
*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :€€€€€€€€€@:€€€€€€€€€@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
И
ц
while_cond_48464
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_48464___redundant_placeholder0-
)while_cond_48464___redundant_placeholder1-
)while_cond_48464___redundant_placeholder2-
)while_cond_48464___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
«
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_48964

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ЪX
э
A__inference_lstm_2_layer_call_and_return_conditional_losses_47714
inputs_0.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resource
identityИҐwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЖ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ц2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Ц   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2э
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€Ц*
shrink_axis_mask2
strided_slice_2≥
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource* 
_output_shapes
:
ЦА*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOp™
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMulЄ
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp¶
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMul_1Ь
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/add±
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp©
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimп
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_4/splitГ
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/SigmoidЗ
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_1И
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/ReluШ
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_1Н
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_1З
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
TensorArrayV2_1/element_shapeЄ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterй
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_47629*
condR
while_cond_47628*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:€€€€€€€€€€€€€€€€€€Ц:::2
whilewhile:_ [
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ц
"
_user_specified_name
inputs/0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
эD
”
A__inference_lstm_2_layer_call_and_return_conditional_losses_45111

inputs
lstm_cell_4_45029
lstm_cell_4_45031
lstm_cell_4_45033
identityИҐ#lstm_cell_4/StatefulPartitionedCallҐwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permД
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ц2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Ц   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2э
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€Ц*
shrink_axis_mask2
strided_slice_2с
#lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_4_45029lstm_cell_4_45031lstm_cell_4_45033*
Tin

2*
Tout
2*M
_output_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_447152%
#lstm_cell_4/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
TensorArrayV2_1/element_shapeЄ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЫ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_4_45029lstm_cell_4_45031lstm_cell_4_45033*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_45042*
condR
while_cond_45041*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeЮ
IdentityIdentitytranspose_1:y:0$^lstm_cell_4/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:€€€€€€€€€€€€€€€€€€Ц:::2J
#lstm_cell_4/StatefulPartitionedCall#lstm_cell_4/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ц
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
п
Ћ
+__inference_lstm_cell_5_layer_call_fn_49177

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2ИҐStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*M
_output_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_453252
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1Т

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
д2
п
while_body_45933
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
,lstm_cell_4_matmul_readvariableop_resource_02
.lstm_cell_4_matmul_1_readvariableop_resource_01
-lstm_cell_4_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resourceИЈ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Ц   23
1TensorArrayV2Read/TensorListGetItem/element_shapeґ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€Ц*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemµ
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_readvariableop_resource_0* 
_output_shapes
:
ЦА*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOpЉ
lstm_cell_4/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMulЇ
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp•
lstm_cell_4/MatMul_1MatMulplaceholder_2+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMul_1Ь
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/add≥
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp©
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimп
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_4/splitГ
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/SigmoidЗ
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_1Е
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/ReluШ
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_1Н
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_1З
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_2Ѕ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3m

Identity_4Identitylstm_cell_4/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_4m

Identity_5Identitylstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"\
+lstm_cell_4_biasadd_readvariableop_resource-lstm_cell_4_biasadd_readvariableop_resource_0"^
,lstm_cell_4_matmul_1_readvariableop_resource.lstm_cell_4_matmul_1_readvariableop_resource_0"Z
*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :€€€€€€€€€@:€€€€€€€€€@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
А
c
D__inference_dropout_5_layer_call_and_return_conditional_losses_46635

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
СX
э
A__inference_lstm_3_layer_call_and_return_conditional_losses_48550
inputs_0.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resource
identityИҐwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЕ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_2≤
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOp™
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMulЄ
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp¶
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMul_1Ь
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/add±
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp©
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dimп
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_5/splitГ
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/SigmoidЗ
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_1И
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/ReluШ
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_1Н
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/add_1З
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Relu_1Ь
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
TensorArrayV2_1/element_shapeЄ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterй
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_48465*
condR
while_cond_48464*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€€€€€€€€€€@:::2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
"
_user_specified_name
inputs/0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
И
ц
while_cond_46297
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_46297___redundant_placeholder0-
)while_cond_46297___redundant_placeholder1-
)while_cond_46297___redundant_placeholder2-
)while_cond_46297___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
хD
”
A__inference_lstm_3_layer_call_and_return_conditional_losses_45853

inputs
lstm_cell_5_45771
lstm_cell_5_45773
lstm_cell_5_45775
identityИҐ#lstm_cell_5/StatefulPartitionedCallҐwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_2с
#lstm_cell_5/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_5_45771lstm_cell_5_45773lstm_cell_5_45775*
Tin

2*
Tout
2*M
_output_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_453582%
#lstm_cell_5/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
TensorArrayV2_1/element_shapeЄ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЫ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_5_45771lstm_cell_5_45773lstm_cell_5_45775*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_45784*
condR
while_cond_45783*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeЪ
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_5/StatefulPartitionedCall^while*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€€€€€€€€€€@:::2J
#lstm_cell_5/StatefulPartitionedCall#lstm_cell_5/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
є$
Р
G__inference_sequential_1_layer_call_and_return_conditional_losses_46746

inputs
lstm_2_46718
lstm_2_46720
lstm_2_46722
lstm_3_46726
lstm_3_46728
lstm_3_46730
dense_2_46734
dense_2_46736
dense_3_46740
dense_3_46742
identityИҐdense_2/StatefulPartitionedCallҐdense_3/StatefulPartitionedCallҐ!dropout_3/StatefulPartitionedCallҐ!dropout_4/StatefulPartitionedCallҐ!dropout_5/StatefulPartitionedCallҐlstm_2/StatefulPartitionedCallҐlstm_3/StatefulPartitionedCallъ
lstm_2/StatefulPartitionedCallStatefulPartitionedCallinputslstm_2_46718lstm_2_46720lstm_2_46722*
Tin
2*
Tout
2*,
_output_shapes
:€€€€€€€€€Ц@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_460182 
lstm_2/StatefulPartitionedCallт
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:€€€€€€€€€Ц@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_462132#
!dropout_3/StatefulPartitionedCallЩ
lstm_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0lstm_3_46726lstm_3_46728lstm_3_46730*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_463832 
lstm_3/StatefulPartitionedCallС
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_465782#
!dropout_4/StatefulPartitionedCallО
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_2_46734dense_2_46736*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_466072!
dense_2/StatefulPartitionedCallТ
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_466352#
!dropout_5/StatefulPartitionedCallО
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_3_46740dense_3_46742*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_466642!
dense_3/StatefulPartitionedCallо
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:€€€€€€€€€ЦЦ::::::::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall:U Q
-
_output_shapes
:€€€€€€€€€ЦЦ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
Њ
О
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_45325

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimњ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@::::O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_namestates:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
д
•
G__inference_sequential_1_layer_call_and_return_conditional_losses_46712
input_5
lstm_2_46684
lstm_2_46686
lstm_2_46688
lstm_3_46692
lstm_3_46694
lstm_3_46696
dense_2_46700
dense_2_46702
dense_3_46706
dense_3_46708
identityИҐdense_2/StatefulPartitionedCallҐdense_3/StatefulPartitionedCallҐlstm_2/StatefulPartitionedCallҐlstm_3/StatefulPartitionedCallы
lstm_2/StatefulPartitionedCallStatefulPartitionedCallinput_5lstm_2_46684lstm_2_46686lstm_2_46688*
Tin
2*
Tout
2*,
_output_shapes
:€€€€€€€€€Ц@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_461712 
lstm_2/StatefulPartitionedCallЏ
dropout_3/PartitionedCallPartitionedCall'lstm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:€€€€€€€€€Ц@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_462182
dropout_3/PartitionedCallС
lstm_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0lstm_3_46692lstm_3_46694lstm_3_46696*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_465362 
lstm_3/StatefulPartitionedCall’
dropout_4/PartitionedCallPartitionedCall'lstm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_465832
dropout_4/PartitionedCallЖ
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_2_46700dense_2_46702*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_466072!
dense_2/StatefulPartitionedCall÷
dropout_5/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_466402
dropout_5/PartitionedCallЖ
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_3_46706dense_3_46708*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_466642!
dense_3/StatefulPartitionedCallВ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:€€€€€€€€€ЦЦ::::::::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall:V R
-
_output_shapes
:€€€€€€€€€ЦЦ
!
_user_specified_name	input_5:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
аW
ы
A__inference_lstm_3_layer_call_and_return_conditional_losses_46536

inputs.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resource
identityИҐwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:Ц€€€€€€€€€@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_2≤
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOp™
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMulЄ
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp¶
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMul_1Ь
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/add±
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp©
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dimп
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_5/splitГ
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/SigmoidЗ
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_1И
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/ReluШ
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_1Н
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/add_1З
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Relu_1Ь
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
TensorArrayV2_1/element_shapeЄ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterй
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_46451*
condR
while_cond_46450*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   22
0TensorArrayV2Stack/TensorListStack/element_shapeй
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:Ц€€€€€€€€€@*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¶
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€Ц@:::2
whilewhile:T P
,
_output_shapes
:€€€€€€€€€Ц@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
И
ц
while_cond_45041
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_45041___redundant_placeholder0-
)while_cond_45041___redundant_placeholder1-
)while_cond_45041___redundant_placeholder2-
)while_cond_45041___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
щ

щ
,__inference_sequential_1_layer_call_fn_47536

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identityИҐStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*,
_read_only_resource_inputs

	
**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_467462
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:€€€€€€€€€ЦЦ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:€€€€€€€€€ЦЦ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
А
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_48912

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
аW
ы
A__inference_lstm_3_layer_call_and_return_conditional_losses_46383

inputs.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resource
identityИҐwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:Ц€€€€€€€€€@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_2≤
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOp™
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMulЄ
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp¶
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMul_1Ь
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/add±
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp©
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dimп
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_5/splitГ
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/SigmoidЗ
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_1И
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/ReluШ
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_1Н
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/add_1З
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Relu_1Ь
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
TensorArrayV2_1/element_shapeЄ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterй
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_46298*
condR
while_cond_46297*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   22
0TensorArrayV2Stack/TensorListStack/element_shapeй
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:Ц€€€€€€€€€@*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¶
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€Ц@:::2
whilewhile:T P
,
_output_shapes
:€€€€€€€€€Ц@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
в2
п
while_body_48465
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
,lstm_cell_5_matmul_readvariableop_resource_02
.lstm_cell_5_matmul_1_readvariableop_resource_01
-lstm_cell_5_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resourceИЈ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   23
1TensorArrayV2Read/TensorListGetItem/element_shapeµ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€@*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemі
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOpЉ
lstm_cell_5/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMulЇ
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp•
lstm_cell_5/MatMul_1MatMulplaceholder_2+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMul_1Ь
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/add≥
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp©
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dimп
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_5/splitГ
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/SigmoidЗ
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_1Е
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/ReluШ
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_1Н
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/add_1З
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Relu_1Ь
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_2Ѕ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3m

Identity_4Identitylstm_cell_5/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_4m

Identity_5Identitylstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"\
+lstm_cell_5_biasadd_readvariableop_resource-lstm_cell_5_biasadd_readvariableop_resource_0"^
,lstm_cell_5_matmul_1_readvariableop_resource.lstm_cell_5_matmul_1_readvariableop_resource_0"Z
*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :€€€€€€€€€@:€€€€€€€€€@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
є 
щ
while_body_45174
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
lstm_cell_4_45198_0
lstm_cell_4_45200_0
lstm_cell_4_45202_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
lstm_cell_4_45198
lstm_cell_4_45200
lstm_cell_4_45202ИҐ#lstm_cell_4/StatefulPartitionedCallЈ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Ц   23
1TensorArrayV2Read/TensorListGetItem/element_shapeґ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€Ц*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemЕ
#lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3lstm_cell_4_45198_0lstm_cell_4_45200_0lstm_cell_4_45202_0*
Tin

2*
Tout
2*M
_output_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_447482%
#lstm_cell_4/StatefulPartitionedCallЎ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder,lstm_cell_4/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1r
IdentityIdentity	add_1:z:0$^lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

IdentityЕ

Identity_1Identitywhile_maximum_iterations$^lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1t

Identity_2Identityadd:z:0$^lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2°

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0$^lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3™

Identity_4Identity,lstm_cell_4/StatefulPartitionedCall:output:1$^lstm_cell_4/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_4™

Identity_5Identity,lstm_cell_4/StatefulPartitionedCall:output:2$^lstm_cell_4/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"(
lstm_cell_4_45198lstm_cell_4_45198_0"(
lstm_cell_4_45200lstm_cell_4_45200_0"(
lstm_cell_4_45202lstm_cell_4_45202_0"$
strided_slice_1strided_slice_1_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :€€€€€€€€€@:€€€€€€€€€@: : :::2J
#lstm_cell_4/StatefulPartitionedCall#lstm_cell_4/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
аW
ы
A__inference_lstm_3_layer_call_and_return_conditional_losses_48878

inputs.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resource
identityИҐwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:Ц€€€€€€€€€@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_2≤
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOp™
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMulЄ
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp¶
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMul_1Ь
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/add±
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp©
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dimп
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_5/splitГ
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/SigmoidЗ
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_1И
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/ReluШ
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_1Н
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/add_1З
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Relu_1Ь
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
TensorArrayV2_1/element_shapeЄ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterй
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_48793*
condR
while_cond_48792*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   22
0TensorArrayV2Stack/TensorListStack/element_shapeй
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:Ц€€€€€€€€€@*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¶
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€Ц@:::2
whilewhile:T P
,
_output_shapes
:€€€€€€€€€Ц@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
л
М
&__inference_lstm_2_layer_call_fn_47878
inputs_0
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_451112
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:€€€€€€€€€€€€€€€€€€Ц:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ц
"
_user_specified_name
inputs/0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
д2
п
while_body_46086
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
,lstm_cell_4_matmul_readvariableop_resource_02
.lstm_cell_4_matmul_1_readvariableop_resource_01
-lstm_cell_4_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resourceИЈ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Ц   23
1TensorArrayV2Read/TensorListGetItem/element_shapeґ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€Ц*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemµ
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_readvariableop_resource_0* 
_output_shapes
:
ЦА*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOpЉ
lstm_cell_4/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMulЇ
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp•
lstm_cell_4/MatMul_1MatMulplaceholder_2+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMul_1Ь
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/add≥
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp©
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimп
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_4/splitГ
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/SigmoidЗ
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_1Е
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/ReluШ
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_1Н
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_1З
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_2Ѕ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3m

Identity_4Identitylstm_cell_4/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_4m

Identity_5Identitylstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"\
+lstm_cell_4_biasadd_readvariableop_resource-lstm_cell_4_biasadd_readvariableop_resource_0"^
,lstm_cell_4_matmul_1_readvariableop_resource.lstm_cell_4_matmul_1_readvariableop_resource_0"Z
*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :€€€€€€€€€@:€€€€€€€€€@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
…
Р
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_49027

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ЦА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimњ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:€€€€€€€€€Ц:€€€€€€€€€@:€€€€€€€€€@::::P L
(
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
И
ц
while_cond_45651
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_45651___redundant_placeholder0-
)while_cond_45651___redundant_placeholder1-
)while_cond_45651___redundant_placeholder2-
)while_cond_45651___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
«
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_46583

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ЪX
э
A__inference_lstm_2_layer_call_and_return_conditional_losses_47867
inputs_0.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resource
identityИҐwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЖ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ц2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Ц   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2э
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€Ц*
shrink_axis_mask2
strided_slice_2≥
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource* 
_output_shapes
:
ЦА*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOp™
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMulЄ
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp¶
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMul_1Ь
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/add±
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp©
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimп
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_4/splitГ
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/SigmoidЗ
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_1И
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/ReluШ
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_1Н
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_1З
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
TensorArrayV2_1/element_shapeЄ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterй
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_47782*
condR
while_cond_47781*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:€€€€€€€€€€€€€€€€€€Ц:::2
whilewhile:_ [
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ц
"
_user_specified_name
inputs/0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Љ$
С
G__inference_sequential_1_layer_call_and_return_conditional_losses_46681
input_5
lstm_2_46194
lstm_2_46196
lstm_2_46198
lstm_3_46559
lstm_3_46561
lstm_3_46563
dense_2_46618
dense_2_46620
dense_3_46675
dense_3_46677
identityИҐdense_2/StatefulPartitionedCallҐdense_3/StatefulPartitionedCallҐ!dropout_3/StatefulPartitionedCallҐ!dropout_4/StatefulPartitionedCallҐ!dropout_5/StatefulPartitionedCallҐlstm_2/StatefulPartitionedCallҐlstm_3/StatefulPartitionedCallы
lstm_2/StatefulPartitionedCallStatefulPartitionedCallinput_5lstm_2_46194lstm_2_46196lstm_2_46198*
Tin
2*
Tout
2*,
_output_shapes
:€€€€€€€€€Ц@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_460182 
lstm_2/StatefulPartitionedCallт
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:€€€€€€€€€Ц@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_462132#
!dropout_3/StatefulPartitionedCallЩ
lstm_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0lstm_3_46559lstm_3_46561lstm_3_46563*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_463832 
lstm_3/StatefulPartitionedCallС
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_465782#
!dropout_4/StatefulPartitionedCallО
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_2_46618dense_2_46620*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_466072!
dense_2/StatefulPartitionedCallТ
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_466352#
!dropout_5/StatefulPartitionedCallО
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_3_46675dense_3_46677*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_466642!
dense_3/StatefulPartitionedCallо
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:€€€€€€€€€ЦЦ::::::::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall:V R
-
_output_shapes
:€€€€€€€€€ЦЦ
!
_user_specified_name	input_5:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
р
E
)__inference_dropout_4_layer_call_fn_48927

inputs
identity†
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_465832
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Ѕ
О
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_44748

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ЦА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimњ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:€€€€€€€€€Ц:€€€€€€€€€@:€€€€€€€€€@::::P L
(
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_namestates:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
©
c
D__inference_dropout_3_layer_call_and_return_conditional_losses_46213

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeє
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/y√
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€Ц@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2

Identity"
identityIdentity:output:0*+
_input_shapes
:€€€€€€€€€Ц@:T P
,
_output_shapes
:€€€€€€€€€Ц@
 
_user_specified_nameinputs
И
ц
while_cond_48639
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_48639___redundant_placeholder0-
)while_cond_48639___redundant_placeholder1-
)while_cond_48639___redundant_placeholder2-
)while_cond_48639___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
∆
Р
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_49127

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimњ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@::::O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
«
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_46640

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
б
§
G__inference_sequential_1_layer_call_and_return_conditional_losses_46802

inputs
lstm_2_46774
lstm_2_46776
lstm_2_46778
lstm_3_46782
lstm_3_46784
lstm_3_46786
dense_2_46790
dense_2_46792
dense_3_46796
dense_3_46798
identityИҐdense_2/StatefulPartitionedCallҐdense_3/StatefulPartitionedCallҐlstm_2/StatefulPartitionedCallҐlstm_3/StatefulPartitionedCallъ
lstm_2/StatefulPartitionedCallStatefulPartitionedCallinputslstm_2_46774lstm_2_46776lstm_2_46778*
Tin
2*
Tout
2*,
_output_shapes
:€€€€€€€€€Ц@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_461712 
lstm_2/StatefulPartitionedCallЏ
dropout_3/PartitionedCallPartitionedCall'lstm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:€€€€€€€€€Ц@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_462182
dropout_3/PartitionedCallС
lstm_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0lstm_3_46782lstm_3_46784lstm_3_46786*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_465362 
lstm_3/StatefulPartitionedCall’
dropout_4/PartitionedCallPartitionedCall'lstm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_465832
dropout_4/PartitionedCallЖ
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_2_46790dense_2_46792*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_466072!
dense_2/StatefulPartitionedCall÷
dropout_5/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_466402
dropout_5/PartitionedCallЖ
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_3_46796dense_3_46798*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_466642!
dense_3/StatefulPartitionedCallВ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:€€€€€€€€€ЦЦ::::::::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall:U Q
-
_output_shapes
:€€€€€€€€€ЦЦ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
И
ц
while_cond_48109
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_48109___redundant_placeholder0-
)while_cond_48109___redundant_placeholder1-
)while_cond_48109___redundant_placeholder2-
)while_cond_48109___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
ь
b
)__inference_dropout_5_layer_call_fn_48969

inputs
identityИҐStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_466352
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ƒ3
†
lstm_3_while_body_47077
lstm_3_while_loop_counter#
lstm_3_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_3_strided_slice_1_0X
Ttensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_00
,lstm_cell_5_matmul_readvariableop_resource_02
.lstm_cell_5_matmul_1_readvariableop_resource_01
-lstm_cell_5_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_3_strided_slice_1V
Rtensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resourceИЈ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   23
1TensorArrayV2Read/TensorListGetItem/element_shapeЉ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemTtensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€@*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemі
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOpЉ
lstm_cell_5/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMulЇ
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp•
lstm_cell_5/MatMul_1MatMulplaceholder_2+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMul_1Ь
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/add≥
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp©
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dimп
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_5/splitГ
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/SigmoidЗ
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_1Е
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/ReluШ
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_1Н
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/add_1З
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Relu_1Ь
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_2Ѕ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/ye
add_1AddV2lstm_3_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identityf

Identity_1Identitylstm_3_while_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3m

Identity_4Identitylstm_cell_5/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_4m

Identity_5Identitylstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"2
lstm_3_strided_slice_1lstm_3_strided_slice_1_0"\
+lstm_cell_5_biasadd_readvariableop_resource-lstm_cell_5_biasadd_readvariableop_resource_0"^
,lstm_cell_5_matmul_1_readvariableop_resource.lstm_cell_5_matmul_1_readvariableop_resource_0"Z
*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_readvariableop_resource_0"™
Rtensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensorTtensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :€€€€€€€€€@:€€€€€€€€€@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
£*
•
__inference__traced_save_49251
file_prefix-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop8
4savev2_lstm_2_lstm_cell_2_kernel_read_readvariableopB
>savev2_lstm_2_lstm_cell_2_recurrent_kernel_read_readvariableop6
2savev2_lstm_2_lstm_cell_2_bias_read_readvariableop8
4savev2_lstm_3_lstm_cell_3_kernel_read_readvariableopB
>savev2_lstm_3_lstm_cell_3_recurrent_kernel_read_readvariableop6
2savev2_lstm_3_lstm_cell_3_bias_read_readvariableop
savev2_1_const

identity_1ИҐMergeV2CheckpointsҐSaveV2ҐSaveV2_1П
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
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_b3052ec62ec141c6ba209aaa75384d10/part2	
Const_1Л
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
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЛ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*Э
valueУBР
B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesЬ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B 2
SaveV2/shape_and_slices±
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop4savev2_lstm_2_lstm_cell_2_kernel_read_readvariableop>savev2_lstm_2_lstm_cell_2_recurrent_kernel_read_readvariableop2savev2_lstm_2_lstm_cell_2_bias_read_readvariableop4savev2_lstm_3_lstm_cell_3_kernel_read_readvariableop>savev2_lstm_3_lstm_cell_3_recurrent_kernel_read_readvariableop2savev2_lstm_3_lstm_cell_3_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
2
2
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardђ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1Ґ
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesѕ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1г
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesђ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*r
_input_shapesa
_: :@::::
ЦА:	@А:А:	@А:	@А:А: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::&"
 
_output_shapes
:
ЦА:%!

_output_shapes
:	@А:!

_output_shapes	
:А:%!

_output_shapes
:	@А:%	!

_output_shapes
:	@А:!


_output_shapes	
:А:

_output_shapes
: 
с
Ћ
+__inference_lstm_cell_4_layer_call_fn_49094

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2ИҐStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*M
_output_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_447482
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1Т

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:€€€€€€€€€Ц:€€€€€€€€€@:€€€€€€€€€@:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
А
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_46578

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
И
ц
while_cond_47781
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_47781___redundant_placeholder0-
)while_cond_47781___redundant_placeholder1-
)while_cond_47781___redundant_placeholder2-
)while_cond_47781___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
И
ц
while_cond_45173
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_45173___redundant_placeholder0-
)while_cond_45173___redundant_placeholder1-
)while_cond_45173___redundant_placeholder2-
)while_cond_45173___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
в2
п
while_body_46298
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
,lstm_cell_5_matmul_readvariableop_resource_02
.lstm_cell_5_matmul_1_readvariableop_resource_01
-lstm_cell_5_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resourceИЈ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   23
1TensorArrayV2Read/TensorListGetItem/element_shapeµ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€@*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemі
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOpЉ
lstm_cell_5/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMulЇ
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp•
lstm_cell_5/MatMul_1MatMulplaceholder_2+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMul_1Ь
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/add≥
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp©
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dimп
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_5/splitГ
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/SigmoidЗ
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_1Е
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/ReluШ
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_1Н
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/add_1З
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Relu_1Ь
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_2Ѕ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3m

Identity_4Identitylstm_cell_5/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_4m

Identity_5Identitylstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"\
+lstm_cell_5_biasadd_readvariableop_resource-lstm_cell_5_biasadd_readvariableop_resource_0"^
,lstm_cell_5_matmul_1_readvariableop_resource.lstm_cell_5_matmul_1_readvariableop_resource_0"Z
*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :€€€€€€€€€@:€€€€€€€€€@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
∞Ў
Ж
G__inference_sequential_1_layer_call_and_return_conditional_losses_47511

inputs5
1lstm_2_lstm_cell_4_matmul_readvariableop_resource7
3lstm_2_lstm_cell_4_matmul_1_readvariableop_resource6
2lstm_2_lstm_cell_4_biasadd_readvariableop_resource5
1lstm_3_lstm_cell_5_matmul_readvariableop_resource7
3lstm_3_lstm_cell_5_matmul_1_readvariableop_resource6
2lstm_3_lstm_cell_5_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identityИҐlstm_2/whileҐlstm_3/whileR
lstm_2/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_2/ShapeВ
lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice/stackЖ
lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_2/strided_slice/stack_1Ж
lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_2/strided_slice/stack_2М
lstm_2/strided_sliceStridedSlicelstm_2/Shape:output:0#lstm_2/strided_slice/stack:output:0%lstm_2/strided_slice/stack_1:output:0%lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_2/strided_slicej
lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_2/zeros/mul/yИ
lstm_2/zeros/mulMullstm_2/strided_slice:output:0lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros/mulm
lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm_2/zeros/Less/yГ
lstm_2/zeros/LessLesslstm_2/zeros/mul:z:0lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros/Lessp
lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_2/zeros/packed/1Я
lstm_2/zeros/packedPacklstm_2/strided_slice:output:0lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_2/zeros/packedm
lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/zeros/ConstС
lstm_2/zerosFilllstm_2/zeros/packed:output:0lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_2/zerosn
lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_2/zeros_1/mul/yО
lstm_2/zeros_1/mulMullstm_2/strided_slice:output:0lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros_1/mulq
lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm_2/zeros_1/Less/yЛ
lstm_2/zeros_1/LessLesslstm_2/zeros_1/mul:z:0lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros_1/Lesst
lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_2/zeros_1/packed/1•
lstm_2/zeros_1/packedPacklstm_2/strided_slice:output:0 lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_2/zeros_1/packedq
lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/zeros_1/ConstЩ
lstm_2/zeros_1Filllstm_2/zeros_1/packed:output:0lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_2/zeros_1Г
lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_2/transpose/permС
lstm_2/transpose	Transposeinputslstm_2/transpose/perm:output:0*
T0*-
_output_shapes
:Ц€€€€€€€€€Ц2
lstm_2/transposed
lstm_2/Shape_1Shapelstm_2/transpose:y:0*
T0*
_output_shapes
:2
lstm_2/Shape_1Ж
lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice_1/stackК
lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_1/stack_1К
lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_1/stack_2Ш
lstm_2/strided_slice_1StridedSlicelstm_2/Shape_1:output:0%lstm_2/strided_slice_1/stack:output:0'lstm_2/strided_slice_1/stack_1:output:0'lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_2/strided_slice_1У
"lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2$
"lstm_2/TensorArrayV2/element_shapeќ
lstm_2/TensorArrayV2TensorListReserve+lstm_2/TensorArrayV2/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_2/TensorArrayV2Ќ
<lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Ц   2>
<lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeФ
.lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_2/transpose:y:0Elstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_2/TensorArrayUnstack/TensorListFromTensorЖ
lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice_2/stackК
lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_2/stack_1К
lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_2/stack_2І
lstm_2/strided_slice_2StridedSlicelstm_2/transpose:y:0%lstm_2/strided_slice_2/stack:output:0'lstm_2/strided_slice_2/stack_1:output:0'lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€Ц*
shrink_axis_mask2
lstm_2/strided_slice_2»
(lstm_2/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp1lstm_2_lstm_cell_4_matmul_readvariableop_resource* 
_output_shapes
:
ЦА*
dtype02*
(lstm_2/lstm_cell_4/MatMul/ReadVariableOp∆
lstm_2/lstm_cell_4/MatMulMatMullstm_2/strided_slice_2:output:00lstm_2/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_2/lstm_cell_4/MatMulЌ
*lstm_2/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp3lstm_2_lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02,
*lstm_2/lstm_cell_4/MatMul_1/ReadVariableOp¬
lstm_2/lstm_cell_4/MatMul_1MatMullstm_2/zeros:output:02lstm_2/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_2/lstm_cell_4/MatMul_1Є
lstm_2/lstm_cell_4/addAddV2#lstm_2/lstm_cell_4/MatMul:product:0%lstm_2/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_2/lstm_cell_4/add∆
)lstm_2/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp2lstm_2_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)lstm_2/lstm_cell_4/BiasAdd/ReadVariableOp≈
lstm_2/lstm_cell_4/BiasAddBiasAddlstm_2/lstm_cell_4/add:z:01lstm_2/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_2/lstm_cell_4/BiasAddv
lstm_2/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_2/lstm_cell_4/ConstК
"lstm_2/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_2/lstm_cell_4/split/split_dimЛ
lstm_2/lstm_cell_4/splitSplit+lstm_2/lstm_cell_4/split/split_dim:output:0#lstm_2/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_2/lstm_cell_4/splitШ
lstm_2/lstm_cell_4/SigmoidSigmoid!lstm_2/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_2/lstm_cell_4/SigmoidЬ
lstm_2/lstm_cell_4/Sigmoid_1Sigmoid!lstm_2/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_2/lstm_cell_4/Sigmoid_1§
lstm_2/lstm_cell_4/mulMul lstm_2/lstm_cell_4/Sigmoid_1:y:0lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_2/lstm_cell_4/mulП
lstm_2/lstm_cell_4/ReluRelu!lstm_2/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_2/lstm_cell_4/Reluі
lstm_2/lstm_cell_4/mul_1Mullstm_2/lstm_cell_4/Sigmoid:y:0%lstm_2/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_2/lstm_cell_4/mul_1©
lstm_2/lstm_cell_4/add_1AddV2lstm_2/lstm_cell_4/mul:z:0lstm_2/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_2/lstm_cell_4/add_1Ь
lstm_2/lstm_cell_4/Sigmoid_2Sigmoid!lstm_2/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_2/lstm_cell_4/Sigmoid_2О
lstm_2/lstm_cell_4/Relu_1Relulstm_2/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_2/lstm_cell_4/Relu_1Є
lstm_2/lstm_cell_4/mul_2Mul lstm_2/lstm_cell_4/Sigmoid_2:y:0'lstm_2/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_2/lstm_cell_4/mul_2Э
$lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2&
$lstm_2/TensorArrayV2_1/element_shape‘
lstm_2/TensorArrayV2_1TensorListReserve-lstm_2/TensorArrayV2_1/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_2/TensorArrayV2_1\
lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_2/timeН
lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2!
lstm_2/while/maximum_iterationsx
lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_2/while/loop_counter“
lstm_2/whileWhile"lstm_2/while/loop_counter:output:0(lstm_2/while/maximum_iterations:output:0lstm_2/time:output:0lstm_2/TensorArrayV2_1:handle:0lstm_2/zeros:output:0lstm_2/zeros_1:output:0lstm_2/strided_slice_1:output:0>lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_2_lstm_cell_4_matmul_readvariableop_resource3lstm_2_lstm_cell_4_matmul_1_readvariableop_resource2lstm_2_lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*#
bodyR
lstm_2_while_body_47260*#
condR
lstm_2_while_cond_47259*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
lstm_2/while√
7lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   29
7lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeЕ
)lstm_2/TensorArrayV2Stack/TensorListStackTensorListStacklstm_2/while:output:3@lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:Ц€€€€€€€€€@*
element_dtype02+
)lstm_2/TensorArrayV2Stack/TensorListStackП
lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
lstm_2/strided_slice_3/stackК
lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_2/strided_slice_3/stack_1К
lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_3/stack_2ƒ
lstm_2/strided_slice_3StridedSlice2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_2/strided_slice_3/stack:output:0'lstm_2/strided_slice_3/stack_1:output:0'lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
lstm_2/strided_slice_3З
lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_2/transpose_1/perm¬
lstm_2/transpose_1	Transpose2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_2/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2
lstm_2/transpose_1t
lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/runtimeГ
dropout_3/IdentityIdentitylstm_2/transpose_1:y:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2
dropout_3/Identityg
lstm_3/ShapeShapedropout_3/Identity:output:0*
T0*
_output_shapes
:2
lstm_3/ShapeВ
lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice/stackЖ
lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_1Ж
lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_2М
lstm_3/strided_sliceStridedSlicelstm_3/Shape:output:0#lstm_3/strided_slice/stack:output:0%lstm_3/strided_slice/stack_1:output:0%lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slicej
lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_3/zeros/mul/yИ
lstm_3/zeros/mulMullstm_3/strided_slice:output:0lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/mulm
lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm_3/zeros/Less/yГ
lstm_3/zeros/LessLesslstm_3/zeros/mul:z:0lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/Lessp
lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_3/zeros/packed/1Я
lstm_3/zeros/packedPacklstm_3/strided_slice:output:0lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros/packedm
lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros/ConstС
lstm_3/zerosFilllstm_3/zeros/packed:output:0lstm_3/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_3/zerosn
lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_3/zeros_1/mul/yО
lstm_3/zeros_1/mulMullstm_3/strided_slice:output:0lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/mulq
lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm_3/zeros_1/Less/yЛ
lstm_3/zeros_1/LessLesslstm_3/zeros_1/mul:z:0lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/Lesst
lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_3/zeros_1/packed/1•
lstm_3/zeros_1/packedPacklstm_3/strided_slice:output:0 lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros_1/packedq
lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros_1/ConstЩ
lstm_3/zeros_1Filllstm_3/zeros_1/packed:output:0lstm_3/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_3/zeros_1Г
lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_3/transpose/perm•
lstm_3/transpose	Transposedropout_3/Identity:output:0lstm_3/transpose/perm:output:0*
T0*,
_output_shapes
:Ц€€€€€€€€€@2
lstm_3/transposed
lstm_3/Shape_1Shapelstm_3/transpose:y:0*
T0*
_output_shapes
:2
lstm_3/Shape_1Ж
lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice_1/stackК
lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_1/stack_1К
lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_1/stack_2Ш
lstm_3/strided_slice_1StridedSlicelstm_3/Shape_1:output:0%lstm_3/strided_slice_1/stack:output:0'lstm_3/strided_slice_1/stack_1:output:0'lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slice_1У
"lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2$
"lstm_3/TensorArrayV2/element_shapeќ
lstm_3/TensorArrayV2TensorListReserve+lstm_3/TensorArrayV2/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_3/TensorArrayV2Ќ
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2>
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeФ
.lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_3/transpose:y:0Elstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_3/TensorArrayUnstack/TensorListFromTensorЖ
lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice_2/stackК
lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_2/stack_1К
lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_2/stack_2¶
lstm_3/strided_slice_2StridedSlicelstm_3/transpose:y:0%lstm_3/strided_slice_2/stack:output:0'lstm_3/strided_slice_2/stack_1:output:0'lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
lstm_3/strided_slice_2«
(lstm_3/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp1lstm_3_lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02*
(lstm_3/lstm_cell_5/MatMul/ReadVariableOp∆
lstm_3/lstm_cell_5/MatMulMatMullstm_3/strided_slice_2:output:00lstm_3/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_3/lstm_cell_5/MatMulЌ
*lstm_3/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp3lstm_3_lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02,
*lstm_3/lstm_cell_5/MatMul_1/ReadVariableOp¬
lstm_3/lstm_cell_5/MatMul_1MatMullstm_3/zeros:output:02lstm_3/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_3/lstm_cell_5/MatMul_1Є
lstm_3/lstm_cell_5/addAddV2#lstm_3/lstm_cell_5/MatMul:product:0%lstm_3/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_3/lstm_cell_5/add∆
)lstm_3/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp2lstm_3_lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)lstm_3/lstm_cell_5/BiasAdd/ReadVariableOp≈
lstm_3/lstm_cell_5/BiasAddBiasAddlstm_3/lstm_cell_5/add:z:01lstm_3/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_3/lstm_cell_5/BiasAddv
lstm_3/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/lstm_cell_5/ConstК
"lstm_3/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_3/lstm_cell_5/split/split_dimЛ
lstm_3/lstm_cell_5/splitSplit+lstm_3/lstm_cell_5/split/split_dim:output:0#lstm_3/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_3/lstm_cell_5/splitШ
lstm_3/lstm_cell_5/SigmoidSigmoid!lstm_3/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_3/lstm_cell_5/SigmoidЬ
lstm_3/lstm_cell_5/Sigmoid_1Sigmoid!lstm_3/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_3/lstm_cell_5/Sigmoid_1§
lstm_3/lstm_cell_5/mulMul lstm_3/lstm_cell_5/Sigmoid_1:y:0lstm_3/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_3/lstm_cell_5/mulП
lstm_3/lstm_cell_5/ReluRelu!lstm_3/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_3/lstm_cell_5/Reluі
lstm_3/lstm_cell_5/mul_1Mullstm_3/lstm_cell_5/Sigmoid:y:0%lstm_3/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_3/lstm_cell_5/mul_1©
lstm_3/lstm_cell_5/add_1AddV2lstm_3/lstm_cell_5/mul:z:0lstm_3/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_3/lstm_cell_5/add_1Ь
lstm_3/lstm_cell_5/Sigmoid_2Sigmoid!lstm_3/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_3/lstm_cell_5/Sigmoid_2О
lstm_3/lstm_cell_5/Relu_1Relulstm_3/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_3/lstm_cell_5/Relu_1Є
lstm_3/lstm_cell_5/mul_2Mul lstm_3/lstm_cell_5/Sigmoid_2:y:0'lstm_3/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_3/lstm_cell_5/mul_2Э
$lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2&
$lstm_3/TensorArrayV2_1/element_shape‘
lstm_3/TensorArrayV2_1TensorListReserve-lstm_3/TensorArrayV2_1/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_3/TensorArrayV2_1\
lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/timeН
lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2!
lstm_3/while/maximum_iterationsx
lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/while/loop_counter“
lstm_3/whileWhile"lstm_3/while/loop_counter:output:0(lstm_3/while/maximum_iterations:output:0lstm_3/time:output:0lstm_3/TensorArrayV2_1:handle:0lstm_3/zeros:output:0lstm_3/zeros_1:output:0lstm_3/strided_slice_1:output:0>lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_3_lstm_cell_5_matmul_readvariableop_resource3lstm_3_lstm_cell_5_matmul_1_readvariableop_resource2lstm_3_lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*#
bodyR
lstm_3_while_body_47410*#
condR
lstm_3_while_cond_47409*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
lstm_3/while√
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   29
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeЕ
)lstm_3/TensorArrayV2Stack/TensorListStackTensorListStacklstm_3/while:output:3@lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:Ц€€€€€€€€€@*
element_dtype02+
)lstm_3/TensorArrayV2Stack/TensorListStackП
lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
lstm_3/strided_slice_3/stackК
lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_3/strided_slice_3/stack_1К
lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_3/stack_2ƒ
lstm_3/strided_slice_3StridedSlice2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_3/strided_slice_3/stack:output:0'lstm_3/strided_slice_3/stack_1:output:0'lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
lstm_3/strided_slice_3З
lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_3/transpose_1/perm¬
lstm_3/transpose_1	Transpose2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_3/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2
lstm_3/transpose_1t
lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/runtimeЗ
dropout_4/IdentityIdentitylstm_3/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_4/Identity•
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_2/MatMul/ReadVariableOp†
dense_2/MatMulMatMuldropout_4/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_2/MatMul§
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp°
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_2/ReluВ
dropout_5/IdentityIdentitydense_2/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€2
dropout_5/Identity•
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp†
dense_3/MatMulMatMuldropout_5/Identity:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_3/MatMul§
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp°
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_3/BiasAddy
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_3/SoftmaxЛ
IdentityIdentitydense_3/Softmax:softmax:0^lstm_2/while^lstm_3/while*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:€€€€€€€€€ЦЦ::::::::::2
lstm_2/whilelstm_2/while2
lstm_3/whilelstm_3/while:U Q
-
_output_shapes
:€€€€€€€€€ЦЦ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
«
Ѓ
lstm_3_while_cond_47076
lstm_3_while_loop_counter#
lstm_3_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_3_strided_slice_14
0lstm_3_while_cond_47076___redundant_placeholder04
0lstm_3_while_cond_47076___redundant_placeholder14
0lstm_3_while_cond_47076___redundant_placeholder24
0lstm_3_while_cond_47076___redundant_placeholder3
identity
_
LessLessplaceholderless_lstm_3_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
д2
п
while_body_48110
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
,lstm_cell_4_matmul_readvariableop_resource_02
.lstm_cell_4_matmul_1_readvariableop_resource_01
-lstm_cell_4_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resourceИЈ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Ц   23
1TensorArrayV2Read/TensorListGetItem/element_shapeґ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€Ц*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemµ
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_readvariableop_resource_0* 
_output_shapes
:
ЦА*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOpЉ
lstm_cell_4/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMulЇ
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp•
lstm_cell_4/MatMul_1MatMulplaceholder_2+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMul_1Ь
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/add≥
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp©
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimп
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_4/splitГ
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/SigmoidЗ
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_1Е
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/ReluШ
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_1Н
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_1З
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_2Ѕ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3m

Identity_4Identitylstm_cell_4/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_4m

Identity_5Identitylstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"\
+lstm_cell_4_biasadd_readvariableop_resource-lstm_cell_4_biasadd_readvariableop_resource_0"^
,lstm_cell_4_matmul_1_readvariableop_resource.lstm_cell_4_matmul_1_readvariableop_resource_0"Z
*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :€€€€€€€€€@:€€€€€€€€€@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
И
ц
while_cond_48792
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_48792___redundant_placeholder0-
)while_cond_48792___redundant_placeholder1-
)while_cond_48792___redundant_placeholder2-
)while_cond_48792___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Р
b
)__inference_dropout_3_layer_call_fn_48239

inputs
identityИҐStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_output_shapes
:€€€€€€€€€Ц@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_462132
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€Ц@2

Identity"
identityIdentity:output:0*+
_input_shapes
:€€€€€€€€€Ц@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€Ц@
 
_user_specified_nameinputs
≈
К
&__inference_lstm_2_layer_call_fn_48206

inputs
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*,
_output_shapes
:€€€€€€€€€Ц@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_460182
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€Ц@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€ЦЦ:::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:€€€€€€€€€ЦЦ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ь

ъ
,__inference_sequential_1_layer_call_fn_46769
input_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identityИҐStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*,
_read_only_resource_inputs

	
**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_467462
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:€€€€€€€€€ЦЦ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
-
_output_shapes
:€€€€€€€€€ЦЦ
!
_user_specified_name	input_5:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
бW
ы
A__inference_lstm_2_layer_call_and_return_conditional_losses_46018

inputs.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resource
identityИҐwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm|
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:Ц€€€€€€€€€Ц2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Ц   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2э
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€Ц*
shrink_axis_mask2
strided_slice_2≥
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource* 
_output_shapes
:
ЦА*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOp™
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMulЄ
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp¶
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMul_1Ь
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/add±
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp©
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimп
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_4/splitГ
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/SigmoidЗ
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_1И
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/ReluШ
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_1Н
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_1З
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
TensorArrayV2_1/element_shapeЄ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterй
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_45933*
condR
while_cond_45932*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   22
0TensorArrayV2Stack/TensorListStack/element_shapeй
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:Ц€€€€€€€€€@*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¶
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimep
IdentityIdentitytranspose_1:y:0^while*
T0*,
_output_shapes
:€€€€€€€€€Ц@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€ЦЦ:::2
whilewhile:U Q
-
_output_shapes
:€€€€€€€€€ЦЦ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
…
Р
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_49060

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ЦА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimњ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:€€€€€€€€€Ц:€€€€€€€€€@:€€€€€€€€€@::::P L
(
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
щ

щ
,__inference_sequential_1_layer_call_fn_47561

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identityИҐStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*,
_read_only_resource_inputs

	
**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_468022
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:€€€€€€€€€ЦЦ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:€€€€€€€€€ЦЦ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
бW
ы
A__inference_lstm_2_layer_call_and_return_conditional_losses_48195

inputs.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resource
identityИҐwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm|
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:Ц€€€€€€€€€Ц2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Ц   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2э
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€Ц*
shrink_axis_mask2
strided_slice_2≥
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource* 
_output_shapes
:
ЦА*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOp™
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMulЄ
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp¶
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/MatMul_1Ь
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/add±
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp©
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimп
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_4/splitГ
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/SigmoidЗ
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_1И
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/ReluШ
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_1Н
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_1З
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
TensorArrayV2_1/element_shapeЄ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterй
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_48110*
condR
while_cond_48109*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   22
0TensorArrayV2Stack/TensorListStack/element_shapeй
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:Ц€€€€€€€€€@*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¶
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ц@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimep
IdentityIdentitytranspose_1:y:0^while*
T0*,
_output_shapes
:€€€€€€€€€Ц@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€ЦЦ:::2
whilewhile:U Q
-
_output_shapes
:€€€€€€€€€ЦЦ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
А
c
D__inference_dropout_5_layer_call_and_return_conditional_losses_48959

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
г
™
B__inference_dense_2_layer_call_and_return_conditional_losses_46607

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@:::O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
хD
”
A__inference_lstm_3_layer_call_and_return_conditional_losses_45721

inputs
lstm_cell_5_45639
lstm_cell_5_45641
lstm_cell_5_45643
identityИҐ#lstm_cell_5/StatefulPartitionedCallҐwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_2с
#lstm_cell_5/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_5_45639lstm_cell_5_45641lstm_cell_5_45643*
Tin

2*
Tout
2*M
_output_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_453252%
#lstm_cell_5/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
TensorArrayV2_1/element_shapeЄ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЫ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_5_45639lstm_cell_5_45641lstm_cell_5_45643*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_45652*
condR
while_cond_45651*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeЪ
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_5/StatefulPartitionedCall^while*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€€€€€€€€€€@:::2J
#lstm_cell_5/StatefulPartitionedCall#lstm_cell_5/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
л
™
B__inference_dense_3_layer_call_and_return_conditional_losses_48985

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€:::O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ћ

с
#__inference_signature_wrapper_46852
input_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identityИҐStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*,
_read_only_resource_inputs

	
**
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__wrapped_model_446422
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:€€€€€€€€€ЦЦ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
-
_output_shapes
:€€€€€€€€€ЦЦ
!
_user_specified_name	input_5:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
ƒ3
†
lstm_3_while_body_47410
lstm_3_while_loop_counter#
lstm_3_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_3_strided_slice_1_0X
Ttensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_00
,lstm_cell_5_matmul_readvariableop_resource_02
.lstm_cell_5_matmul_1_readvariableop_resource_01
-lstm_cell_5_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_3_strided_slice_1V
Rtensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resourceИЈ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   23
1TensorArrayV2Read/TensorListGetItem/element_shapeЉ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemTtensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€@*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemі
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOpЉ
lstm_cell_5/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMulЇ
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp•
lstm_cell_5/MatMul_1MatMulplaceholder_2+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/MatMul_1Ь
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/add≥
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp©
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dimп
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*
	num_split2
lstm_cell_5/splitГ
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/SigmoidЗ
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_1Е
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/ReluШ
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_1Н
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/add_1З
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/Relu_1Ь
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_5/mul_2Ѕ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/ye
add_1AddV2lstm_3_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identityf

Identity_1Identitylstm_3_while_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3m

Identity_4Identitylstm_cell_5/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_4m

Identity_5Identitylstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"2
lstm_3_strided_slice_1lstm_3_strided_slice_1_0"\
+lstm_cell_5_biasadd_readvariableop_resource-lstm_cell_5_biasadd_readvariableop_resource_0"^
,lstm_cell_5_matmul_1_readvariableop_resource.lstm_cell_5_matmul_1_readvariableop_resource_0"Z
*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_readvariableop_resource_0"™
Rtensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensorTtensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :€€€€€€€€€@:€€€€€€€€€@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: "ѓL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*∞
serving_defaultЬ
A
input_56
serving_default_input_5:0€€€€€€€€€ЦЦ;
dense_30
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:°≤
Р<
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
	optimizer
	
signatures

trainable_variables
regularization_losses
	variables
	keras_api
t_default_save_signature
*u&call_and_return_all_conditional_losses
v__call__"В9
_tf_keras_sequentialг8{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_1", "layers": [{"class_name": "LSTM", "config": {"name": "lstm_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 150, 150]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "LSTM", "config": {"name": "lstm_3", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 150, 150]}}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 150]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 150, 150]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "LSTM", "config": {"name": "lstm_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 150, 150]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "LSTM", "config": {"name": "lstm_3", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 150, 150]}}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 9.999999747378752e-06, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Ы
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
*w&call_and_return_all_conditional_losses
x__call__"т

_tf_keras_rnn_layer‘
{"class_name": "LSTM", "name": "lstm_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 150, 150]}, "stateful": false, "config": {"name": "lstm_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 150, 150]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 150]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 150, 150]}}
¬
	variables
regularization_losses
trainable_variables
	keras_api
*y&call_and_return_all_conditional_losses
z__call__"≥
_tf_keras_layerЩ{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
Ы
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
*{&call_and_return_all_conditional_losses
|__call__"т	
_tf_keras_rnn_layer‘	{"class_name": "LSTM", "name": "lstm_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "lstm_3", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 64]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 150, 64]}}
¬
	variables
regularization_losses
 trainable_variables
!	keras_api
*}&call_and_return_all_conditional_losses
~__call__"≥
_tf_keras_layerЩ{"class_name": "Dropout", "name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
ќ

"kernel
#bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
*&call_and_return_all_conditional_losses
А__call__"®
_tf_keras_layerО{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ƒ
(	variables
)regularization_losses
*trainable_variables
+	keras_api
+Б&call_and_return_all_conditional_losses
В__call__"≥
_tf_keras_layerЩ{"class_name": "Dropout", "name": "dropout_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
—

,kernel
-bias
.	variables
/regularization_losses
0trainable_variables
1	keras_api
+Г&call_and_return_all_conditional_losses
Д__call__"™
_tf_keras_layerР{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
"
	optimizer
-
Еserving_default"
signature_map
f
20
31
42
53
64
75
"6
#7
,8
-9"
trackable_list_wrapper
 "
trackable_list_wrapper
f
20
31
42
53
64
75
"6
#7
,8
-9"
trackable_list_wrapper
 
8non_trainable_variables

trainable_variables

9layers
regularization_losses
:metrics
;layer_metrics
	variables
<layer_regularization_losses
v__call__
t_default_save_signature
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
З

2kernel
3recurrent_kernel
4bias
=	variables
>regularization_losses
?trainable_variables
@	keras_api
+Ж&call_and_return_all_conditional_losses
З__call__" 
_tf_keras_layer∞{"class_name": "LSTMCell", "name": "lstm_cell_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "lstm_cell_4", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
5
20
31
42"
trackable_list_wrapper
5
20
31
42"
trackable_list_wrapper
 "
trackable_list_wrapper
є
Anon_trainable_variables
trainable_variables

Blayers
	variables
regularization_losses
Cmetrics
Dlayer_metrics

Estates
Flayer_regularization_losses
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
Gnon_trainable_variables

Hlayers
	variables
regularization_losses
Imetrics
Jlayer_metrics
trainable_variables
Klayer_regularization_losses
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
З

5kernel
6recurrent_kernel
7bias
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
+И&call_and_return_all_conditional_losses
Й__call__" 
_tf_keras_layer∞{"class_name": "LSTMCell", "name": "lstm_cell_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "lstm_cell_5", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
5
50
61
72"
trackable_list_wrapper
5
50
61
72"
trackable_list_wrapper
 "
trackable_list_wrapper
є
Pnon_trainable_variables
trainable_variables

Qlayers
	variables
regularization_losses
Rmetrics
Slayer_metrics

Tstates
Ulayer_regularization_losses
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
Vnon_trainable_variables

Wlayers
	variables
regularization_losses
Xmetrics
Ylayer_metrics
 trainable_variables
Zlayer_regularization_losses
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_2/kernel
:2dense_2/bias
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
Ѓ
[non_trainable_variables

\layers
$	variables
%regularization_losses
]metrics
^layer_metrics
&trainable_variables
_layer_regularization_losses
А__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
`non_trainable_variables

alayers
(	variables
)regularization_losses
bmetrics
clayer_metrics
*trainable_variables
dlayer_regularization_losses
В__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
 :2dense_3/kernel
:2dense_3/bias
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
∞
enon_trainable_variables

flayers
.	variables
/regularization_losses
gmetrics
hlayer_metrics
0trainable_variables
ilayer_regularization_losses
Д__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
-:+
ЦА2lstm_2/lstm_cell_2/kernel
6:4	@А2#lstm_2/lstm_cell_2/recurrent_kernel
&:$А2lstm_2/lstm_cell_2/bias
,:*	@А2lstm_3/lstm_cell_3/kernel
6:4	@А2#lstm_3/lstm_cell_3/recurrent_kernel
&:$А2lstm_3/lstm_cell_3/bias
 "
trackable_list_wrapper
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
trackable_dict_wrapper
 "
trackable_list_wrapper
5
20
31
42"
trackable_list_wrapper
 "
trackable_list_wrapper
5
20
31
42"
trackable_list_wrapper
∞
jnon_trainable_variables

klayers
=	variables
>regularization_losses
lmetrics
mlayer_metrics
?trainable_variables
nlayer_regularization_losses
З__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
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
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
50
61
72"
trackable_list_wrapper
 "
trackable_list_wrapper
5
50
61
72"
trackable_list_wrapper
∞
onon_trainable_variables

players
L	variables
Mregularization_losses
qmetrics
rlayer_metrics
Ntrainable_variables
slayer_regularization_losses
Й__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
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
д2б
 __inference__wrapped_model_44642Љ
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *,Ґ)
'К$
input_5€€€€€€€€€ЦЦ
к2з
G__inference_sequential_1_layer_call_and_return_conditional_losses_46712
G__inference_sequential_1_layer_call_and_return_conditional_losses_47511
G__inference_sequential_1_layer_call_and_return_conditional_losses_47192
G__inference_sequential_1_layer_call_and_return_conditional_losses_46681ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ю2ы
,__inference_sequential_1_layer_call_fn_47536
,__inference_sequential_1_layer_call_fn_47561
,__inference_sequential_1_layer_call_fn_46825
,__inference_sequential_1_layer_call_fn_46769ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
з2д
A__inference_lstm_2_layer_call_and_return_conditional_losses_48042
A__inference_lstm_2_layer_call_and_return_conditional_losses_48195
A__inference_lstm_2_layer_call_and_return_conditional_losses_47714
A__inference_lstm_2_layer_call_and_return_conditional_losses_47867’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ы2ш
&__inference_lstm_2_layer_call_fn_48217
&__inference_lstm_2_layer_call_fn_48206
&__inference_lstm_2_layer_call_fn_47889
&__inference_lstm_2_layer_call_fn_47878’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
∆2√
D__inference_dropout_3_layer_call_and_return_conditional_losses_48234
D__inference_dropout_3_layer_call_and_return_conditional_losses_48229і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Р2Н
)__inference_dropout_3_layer_call_fn_48239
)__inference_dropout_3_layer_call_fn_48244і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
з2д
A__inference_lstm_3_layer_call_and_return_conditional_losses_48550
A__inference_lstm_3_layer_call_and_return_conditional_losses_48397
A__inference_lstm_3_layer_call_and_return_conditional_losses_48878
A__inference_lstm_3_layer_call_and_return_conditional_losses_48725’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ы2ш
&__inference_lstm_3_layer_call_fn_48889
&__inference_lstm_3_layer_call_fn_48561
&__inference_lstm_3_layer_call_fn_48900
&__inference_lstm_3_layer_call_fn_48572’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
∆2√
D__inference_dropout_4_layer_call_and_return_conditional_losses_48917
D__inference_dropout_4_layer_call_and_return_conditional_losses_48912і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Р2Н
)__inference_dropout_4_layer_call_fn_48927
)__inference_dropout_4_layer_call_fn_48922і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
м2й
B__inference_dense_2_layer_call_and_return_conditional_losses_48938Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_dense_2_layer_call_fn_48947Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
∆2√
D__inference_dropout_5_layer_call_and_return_conditional_losses_48959
D__inference_dropout_5_layer_call_and_return_conditional_losses_48964і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Р2Н
)__inference_dropout_5_layer_call_fn_48969
)__inference_dropout_5_layer_call_fn_48974і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
м2й
B__inference_dense_3_layer_call_and_return_conditional_losses_48985Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_dense_3_layer_call_fn_48994Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
2B0
#__inference_signature_wrapper_46852input_5
‘2—
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_49060
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_49027Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ю2Ы
+__inference_lstm_cell_4_layer_call_fn_49077
+__inference_lstm_cell_4_layer_call_fn_49094Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
‘2—
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_49127
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_49160Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ю2Ы
+__inference_lstm_cell_5_layer_call_fn_49194
+__inference_lstm_cell_5_layer_call_fn_49177Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 Ы
 __inference__wrapped_model_44642w
234567"#,-6Ґ3
,Ґ)
'К$
input_5€€€€€€€€€ЦЦ
™ "1™.
,
dense_3!К
dense_3€€€€€€€€€Ґ
B__inference_dense_2_layer_call_and_return_conditional_losses_48938\"#/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€
Ъ z
'__inference_dense_2_layer_call_fn_48947O"#/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€Ґ
B__inference_dense_3_layer_call_and_return_conditional_losses_48985\,-/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ z
'__inference_dense_3_layer_call_fn_48994O,-/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€Ѓ
D__inference_dropout_3_layer_call_and_return_conditional_losses_48229f8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€Ц@
p
™ "*Ґ'
 К
0€€€€€€€€€Ц@
Ъ Ѓ
D__inference_dropout_3_layer_call_and_return_conditional_losses_48234f8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€Ц@
p 
™ "*Ґ'
 К
0€€€€€€€€€Ц@
Ъ Ж
)__inference_dropout_3_layer_call_fn_48239Y8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€Ц@
p
™ "К€€€€€€€€€Ц@Ж
)__inference_dropout_3_layer_call_fn_48244Y8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€Ц@
p 
™ "К€€€€€€€€€Ц@§
D__inference_dropout_4_layer_call_and_return_conditional_losses_48912\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "%Ґ"
К
0€€€€€€€€€@
Ъ §
D__inference_dropout_4_layer_call_and_return_conditional_losses_48917\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ |
)__inference_dropout_4_layer_call_fn_48922O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "К€€€€€€€€€@|
)__inference_dropout_4_layer_call_fn_48927O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "К€€€€€€€€€@§
D__inference_dropout_5_layer_call_and_return_conditional_losses_48959\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p
™ "%Ґ"
К
0€€€€€€€€€
Ъ §
D__inference_dropout_5_layer_call_and_return_conditional_losses_48964\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p 
™ "%Ґ"
К
0€€€€€€€€€
Ъ |
)__inference_dropout_5_layer_call_fn_48969O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p
™ "К€€€€€€€€€|
)__inference_dropout_5_layer_call_fn_48974O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p 
™ "К€€€€€€€€€—
A__inference_lstm_2_layer_call_and_return_conditional_losses_47714Л234PҐM
FҐC
5Ъ2
0К-
inputs/0€€€€€€€€€€€€€€€€€€Ц

 
p

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€@
Ъ —
A__inference_lstm_2_layer_call_and_return_conditional_losses_47867Л234PҐM
FҐC
5Ъ2
0К-
inputs/0€€€€€€€€€€€€€€€€€€Ц

 
p 

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€@
Ъ є
A__inference_lstm_2_layer_call_and_return_conditional_losses_48042t234AҐ>
7Ґ4
&К#
inputs€€€€€€€€€ЦЦ

 
p

 
™ "*Ґ'
 К
0€€€€€€€€€Ц@
Ъ є
A__inference_lstm_2_layer_call_and_return_conditional_losses_48195t234AҐ>
7Ґ4
&К#
inputs€€€€€€€€€ЦЦ

 
p 

 
™ "*Ґ'
 К
0€€€€€€€€€Ц@
Ъ ®
&__inference_lstm_2_layer_call_fn_47878~234PҐM
FҐC
5Ъ2
0К-
inputs/0€€€€€€€€€€€€€€€€€€Ц

 
p

 
™ "%К"€€€€€€€€€€€€€€€€€€@®
&__inference_lstm_2_layer_call_fn_47889~234PҐM
FҐC
5Ъ2
0К-
inputs/0€€€€€€€€€€€€€€€€€€Ц

 
p 

 
™ "%К"€€€€€€€€€€€€€€€€€€@С
&__inference_lstm_2_layer_call_fn_48206g234AҐ>
7Ґ4
&К#
inputs€€€€€€€€€ЦЦ

 
p

 
™ "К€€€€€€€€€Ц@С
&__inference_lstm_2_layer_call_fn_48217g234AҐ>
7Ґ4
&К#
inputs€€€€€€€€€ЦЦ

 
p 

 
™ "К€€€€€€€€€Ц@¬
A__inference_lstm_3_layer_call_and_return_conditional_losses_48397}567OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€@

 
p

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ¬
A__inference_lstm_3_layer_call_and_return_conditional_losses_48550}567OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€@

 
p 

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ≥
A__inference_lstm_3_layer_call_and_return_conditional_losses_48725n567@Ґ=
6Ґ3
%К"
inputs€€€€€€€€€Ц@

 
p

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ≥
A__inference_lstm_3_layer_call_and_return_conditional_losses_48878n567@Ґ=
6Ґ3
%К"
inputs€€€€€€€€€Ц@

 
p 

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ Ъ
&__inference_lstm_3_layer_call_fn_48561p567OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€@

 
p

 
™ "К€€€€€€€€€@Ъ
&__inference_lstm_3_layer_call_fn_48572p567OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€@

 
p 

 
™ "К€€€€€€€€€@Л
&__inference_lstm_3_layer_call_fn_48889a567@Ґ=
6Ґ3
%К"
inputs€€€€€€€€€Ц@

 
p

 
™ "К€€€€€€€€€@Л
&__inference_lstm_3_layer_call_fn_48900a567@Ґ=
6Ґ3
%К"
inputs€€€€€€€€€Ц@

 
p 

 
™ "К€€€€€€€€€@…
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_49027ю234БҐ~
wҐt
!К
inputs€€€€€€€€€Ц
KҐH
"К
states/0€€€€€€€€€@
"К
states/1€€€€€€€€€@
p
™ "sҐp
iҐf
К
0/0€€€€€€€€€@
EЪB
К
0/1/0€€€€€€€€€@
К
0/1/1€€€€€€€€€@
Ъ …
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_49060ю234БҐ~
wҐt
!К
inputs€€€€€€€€€Ц
KҐH
"К
states/0€€€€€€€€€@
"К
states/1€€€€€€€€€@
p 
™ "sҐp
iҐf
К
0/0€€€€€€€€€@
EЪB
К
0/1/0€€€€€€€€€@
К
0/1/1€€€€€€€€€@
Ъ Ю
+__inference_lstm_cell_4_layer_call_fn_49077о234БҐ~
wҐt
!К
inputs€€€€€€€€€Ц
KҐH
"К
states/0€€€€€€€€€@
"К
states/1€€€€€€€€€@
p
™ "cҐ`
К
0€€€€€€€€€@
AЪ>
К
1/0€€€€€€€€€@
К
1/1€€€€€€€€€@Ю
+__inference_lstm_cell_4_layer_call_fn_49094о234БҐ~
wҐt
!К
inputs€€€€€€€€€Ц
KҐH
"К
states/0€€€€€€€€€@
"К
states/1€€€€€€€€€@
p 
™ "cҐ`
К
0€€€€€€€€€@
AЪ>
К
1/0€€€€€€€€€@
К
1/1€€€€€€€€€@»
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_49127э567АҐ}
vҐs
 К
inputs€€€€€€€€€@
KҐH
"К
states/0€€€€€€€€€@
"К
states/1€€€€€€€€€@
p
™ "sҐp
iҐf
К
0/0€€€€€€€€€@
EЪB
К
0/1/0€€€€€€€€€@
К
0/1/1€€€€€€€€€@
Ъ »
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_49160э567АҐ}
vҐs
 К
inputs€€€€€€€€€@
KҐH
"К
states/0€€€€€€€€€@
"К
states/1€€€€€€€€€@
p 
™ "sҐp
iҐf
К
0/0€€€€€€€€€@
EЪB
К
0/1/0€€€€€€€€€@
К
0/1/1€€€€€€€€€@
Ъ Э
+__inference_lstm_cell_5_layer_call_fn_49177н567АҐ}
vҐs
 К
inputs€€€€€€€€€@
KҐH
"К
states/0€€€€€€€€€@
"К
states/1€€€€€€€€€@
p
™ "cҐ`
К
0€€€€€€€€€@
AЪ>
К
1/0€€€€€€€€€@
К
1/1€€€€€€€€€@Э
+__inference_lstm_cell_5_layer_call_fn_49194н567АҐ}
vҐs
 К
inputs€€€€€€€€€@
KҐH
"К
states/0€€€€€€€€€@
"К
states/1€€€€€€€€€@
p 
™ "cҐ`
К
0€€€€€€€€€@
AЪ>
К
1/0€€€€€€€€€@
К
1/1€€€€€€€€€@Њ
G__inference_sequential_1_layer_call_and_return_conditional_losses_46681s
234567"#,->Ґ;
4Ґ1
'К$
input_5€€€€€€€€€ЦЦ
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Њ
G__inference_sequential_1_layer_call_and_return_conditional_losses_46712s
234567"#,->Ґ;
4Ґ1
'К$
input_5€€€€€€€€€ЦЦ
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ љ
G__inference_sequential_1_layer_call_and_return_conditional_losses_47192r
234567"#,-=Ґ:
3Ґ0
&К#
inputs€€€€€€€€€ЦЦ
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ љ
G__inference_sequential_1_layer_call_and_return_conditional_losses_47511r
234567"#,-=Ґ:
3Ґ0
&К#
inputs€€€€€€€€€ЦЦ
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ц
,__inference_sequential_1_layer_call_fn_46769f
234567"#,->Ґ;
4Ґ1
'К$
input_5€€€€€€€€€ЦЦ
p

 
™ "К€€€€€€€€€Ц
,__inference_sequential_1_layer_call_fn_46825f
234567"#,->Ґ;
4Ґ1
'К$
input_5€€€€€€€€€ЦЦ
p 

 
™ "К€€€€€€€€€Х
,__inference_sequential_1_layer_call_fn_47536e
234567"#,-=Ґ:
3Ґ0
&К#
inputs€€€€€€€€€ЦЦ
p

 
™ "К€€€€€€€€€Х
,__inference_sequential_1_layer_call_fn_47561e
234567"#,-=Ґ:
3Ґ0
&К#
inputs€€€€€€€€€ЦЦ
p 

 
™ "К€€€€€€€€€™
#__inference_signature_wrapper_46852В
234567"#,-AҐ>
Ґ 
7™4
2
input_5'К$
input_5€€€€€€€€€ЦЦ"1™.
,
dense_3!К
dense_3€€€€€€€€€
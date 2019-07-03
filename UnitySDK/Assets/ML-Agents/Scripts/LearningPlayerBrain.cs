#define ENABLE_BARRACUDA

using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Serialization;
using System.Linq;
using Barracuda;
using MLAgents.InferenceBrain;
using UnityEngine.Profiling;
using Tensor = MLAgents.InferenceBrain.Tensor;

namespace MLAgents
{

    
    /// <summary>
    /// The Learning Brain works differently if you are training it or not.
    /// When training your Agents, drag the Learning Brain to the Academy's BroadcastHub and check
    /// the checkbox Control. When using a pretrained model, just drag the Model file into the
    /// Model property of the Learning Brain.
    /// The property model corresponds to the Model currently attached to the Brain. Before
    /// being used, a call to ReloadModel is required.
    /// When the Learning Brain is not training, it uses a TensorFlow model to make decisions.
    /// The Proximal Policy Optimization (PPO) and Behavioral Cloning algorithms included with
    /// the ML-Agents SDK produce trained TensorFlow models that you can use with the
    /// Learning Brain.
    /// </summary>
    [CreateAssetMenu(fileName = "NewLearningPlayerBrain", menuName = "ML-Agents/Learning Player Brain")]
    public class LearningPlayerBrain : LearningBrain
    {
//         private TensorGenerator _tensorGenerator;
//         private TensorApplier _tensorApplier;
// #if ENABLE_TENSORFLOW
//         public TextAsset model;
//         private ModelParamLoader _modelParamLoader;
//         private TFSharpInferenceEngine _engine;
// #elif ENABLE_BARRACUDA 
//         public NNModel model;
//         private Model _barracudaModel;
//         private IWorker _engine;
//         private bool _verbose = false;
        
//         private BarracudaModelParamLoader _modelParamLoader;
//         private string[] _outputNames;
// #endif
//         [Tooltip("Inference execution device. CPU is the fastest option for most of ML Agents models. " +
//                  "(This field is not applicable for training).")]
//         public InferenceDevice inferenceDevice = InferenceDevice.CPU;
        
//         private IReadOnlyList<Tensor> _inferenceInputs;
//         private IReadOnlyList<Tensor> _inferenceOutputs;

//         [NonSerialized]
//         private bool _isControlled;

        /// PLAYER BRAIN TSUFF
        [System.Serializable]
        public struct DiscretePlayerAction
        {
            public KeyCode key;
            public int branchIndex;
            public int value;
        }

        [System.Serializable]
        public struct KeyContinuousPlayerAction
        {
            public KeyCode key;
            public int index;
            public float value;
        }
        
        [System.Serializable]
        public struct AxisContinuousPlayerAction
        {
            public string axis;
            public int index;
            public float scale;
        }

        [SerializeField]
        [FormerlySerializedAs("continuousPlayerActions")]
        [Tooltip("The list of keys and the value they correspond to for continuous control.")]
        /// Contains the mapping from input to continuous actions
        public KeyContinuousPlayerAction[] keyContinuousPlayerActions;
        
        [SerializeField]
        [Tooltip("The list of axis actions.")]
        /// Contains the mapping from input to continuous actions
        public AxisContinuousPlayerAction[] axisContinuousPlayerActions;
        
        [SerializeField]
        [Tooltip("The list of keys and the value they correspond to for discrete control.")]
        /// Contains the mapping from input to discrete actions
        public DiscretePlayerAction[] discretePlayerActions;

        [SerializeField]
        [Tooltip("Whether or not we should allow the player to control it.")]
        public bool isControllable = false;

        protected override void DecideAction()
        {
            bool wasPlayerControlled = false;
                if (brainParameters.vectorActionSpaceType == SpaceType.continuous)
                {
                    foreach (Agent agent in agentInfos.Keys)
                    {
                        if(agent.isControllable)
                        {
                            var action = new float[brainParameters.vectorActionSize[0]];
                            foreach (KeyContinuousPlayerAction cha in keyContinuousPlayerActions)
                            {
                                if (Input.GetKey(cha.key))
                                {
                                    action[cha.index] = cha.value;
                                    wasPlayerControlled = true;
                                }
                            }
                            foreach (AxisContinuousPlayerAction axisAction in axisContinuousPlayerActions)
                            {
                                var axisValue = Input.GetAxis(axisAction.axis);
                                axisValue *= axisAction.scale;
                                if (Mathf.Abs(axisValue) > 0.0001)
                                {
                                    action[axisAction.index] = axisValue;
                                    wasPlayerControlled = true;
                                }
                            }
                            if (wasPlayerControlled){
                                agent.UpdateVectorAction(action); 
                                agent.SetIsDemonstration(true);
                            }
                        }
                    } 
                }
                else
                {
                    foreach (Agent agent in agentInfos.Keys)
                    {
                        if(agent.isControllable)
                        {
                            var action = new float[brainParameters.vectorActionSize.Length];
                            foreach (DiscretePlayerAction dha in discretePlayerActions)
                            {
                                if (Input.GetKey(dha.key))
                                {
                                    action[dha.branchIndex] = (float) dha.value;
                                    wasPlayerControlled = true;
                                }
                            }
                            if (wasPlayerControlled){
                                agent.UpdateVectorAction(action);
                                agent.SetIsDemonstration(true);
                            }
                        }
                    }
                }

            // Adjust timescale based on keypress
            if(Input.anyKey)
            {
                Time.timeScale = 0.3f;
                agentInfos.Clear();
                return;
            }
            else{
                Time.timeScale = 15;
            }
        
            if (_isControlled )
            {        
                agentInfos.Clear();
                return;
            }
            var currentBatchSize = agentInfos.Count();
            if (currentBatchSize == 0)
            {
                return;
            }
#if ENABLE_TENSORFLOW
            if (_engine == null)
            {
                Debug.LogError($"No model was present for the Brain {name}.");
                return;
            }
            // Prepare the input tensors to be feed into the engine
            _tensorGenerator.GenerateTensors(_inferenceInputs, currentBatchSize, agentInfos);
            
            // Prepare the output tensors to be feed into the engine
            _tensorGenerator.GenerateTensors(_inferenceOutputs, currentBatchSize, agentInfos);

            // Execute the Model
            Profiler.BeginSample($"MLAgents.{name}.ExecuteGraph");
            _engine.ExecuteGraph(_inferenceInputs, _inferenceOutputs);
            Profiler.EndSample();

            // Update the outputs
            _tensorApplier.ApplyTensors(_inferenceOutputs, agentInfos);
#elif ENABLE_BARRACUDA
            if (_engine == null)
            {
                Debug.LogError($"No model was present for the Brain {name}.");
                return;
            }
            
            // Prepare the input tensors to be feed into the engine
            _tensorGenerator.GenerateTensors(_inferenceInputs, currentBatchSize, agentInfos);

            var inputs = PrepareBarracudaInputs(_inferenceInputs);

            // Execute the Model
            Profiler.BeginSample($"MLAgents.{name}.ExecuteGraph");
            _engine.Execute(inputs);
            Profiler.EndSample();

            _inferenceOutputs = FetchBarracudaOutputs(_outputNames);
            CleanupBarracudaState(inputs);

            // Update the outputs
            _tensorApplier.ApplyTensors(_inferenceOutputs, agentInfos);
#else
            if (agentInfos.Count > 0)
            {
                Debug.LogError(string.Format(
                    "The brain {0} was set to inference mode but the Tensorflow library is not " +
                    "present in the Unity project.",
                    name));
            }
#endif
            agentInfos.Clear();
        }
        
    }
}

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System.Linq;

namespace WeArt.Components
{
    public class RetargetingController : MonoBehaviour
    {
        public enum ControlMode { Network, Dataset, WeArt}

        [Header("Control Mode")]
        public ControlMode inputSource = ControlMode.Network;

        [Header("Dataset Configuration")]
        private TextAsset datasetFile;
        public string FileName = "data.csv";
        public float frameRate = 30f;
        public int initialFrame = 72;
        public int N_of_frames = 40; 
        private bool isPlayingDataset = false;

        [Header("Human Hand Mapping")]
        public Transform h_palm;
        public Transform[] h_joints; 
        
        public Transform[] h_refPoints;     // ph
        private Vector3[] _h_prevPos;

        [Header("WeArt Thimbles")]
        [SerializeField] private WeArtThimbleTrackingObject thumbThimble;
        [SerializeField] private WeArtThimbleTrackingObject indexThimble;
        [SerializeField] private WeArtThimbleTrackingObject middleThimble;

        [Header("Robot configuration")]
        public Transform r_palm;
        public Transform[] r_joints;        // qr
        public Kinematics.JointType[] r_jointTypes;
        public Transform[] r_refPoints;     // pr

        [Header("Virtual spheres")]
        [SerializeField] private VirtualSphere h_sphere;
        [SerializeField] private VirtualSphere r_sphere;

        private ArticulationBody[] _r_bodies; 
        [Header("Physics Config")]
        public float stiffness = 20000000f; 
        public float driveDamping = 80000f;
        public float forceLimit = 5000f;
        public float linkMass = 0.001f;

        [Header("Dynamics")]
        public float velocityGain = 0.3f;
        public double damping = 0.01;

        [Header("Redundancy resolution")]
        public bool useNullSpace = true;
        public float nullSpaceGain = 0.5f;
        public Vector3[] limitsMin;
        public Vector3[] limitsMax;

        [Header("Network")]
        public string ipAddress = "127.0.0.1";
        public int port = 65432;

        private int _dof;
        
        // network state
        private TcpClient _client;
        private NetworkStream _stream;
        private Thread _netThread;
        private float[] _synergyInput = new float[4];
        private Vector<double> _incomingAngles;
        private volatile bool _hasData = false;
        private volatile bool _active = true;
        private readonly object _lock = new object();

        // dataset state
        private List<float[]> _recordedFrames = new List<float[]>();
        private float _playbackTime = 0f;
        private int _currentFrameIndex = 0;

        void Start()
        {
            // setup physics
            _r_bodies = new ArticulationBody[r_joints.Length];
            for (int i = 0; i < r_joints.Length; i++)
            {
                if (r_joints[i] != null)
                {
                    _r_bodies[i] = r_joints[i].GetComponent<ArticulationBody>();
                    if (_r_bodies[i] != null) ConfigureBody(_r_bodies[i]);
                }
            }

            _dof = Kinematics.GetDoF(r_joints, r_jointTypes);
            _incomingAngles = Vector<double>.Build.Dense(45);
            
            // setup human reference tracking
            _h_prevPos = new Vector3[h_refPoints.Length];
            for (int i=0; i < h_refPoints.Length; i++)
                if (h_refPoints[i] != null) _h_prevPos[i] = h_palm.InverseTransformPoint(h_refPoints[i].position);

            AssignSphereRefs();

            // setup dataset and first pose
            LoadTestDataset();
            if (inputSource == ControlMode.Dataset && _recordedFrames.Count > 0)
                ApplyFrameToInput(0);
        
            // setup network
            _netThread = new Thread(NetworkLoop) { IsBackground = true };
            _netThread.Start();
        }

        private void LoadTestDataset()
        {
            string fullPath = Path.Combine(Application.streamingAssetsPath, FileName);

            if (!File.Exists(fullPath)) { Debug.LogError("CSV not found: " + fullPath); return; }

            _recordedFrames.Clear();
            string[] lines = File.ReadAllLines(fullPath);
            int rowsRead = 0;
            for (int i = initialFrame; i < lines.Length; i++) 
            {
                if (N_of_frames > 0 && rowsRead >= N_of_frames) break;
                if (string.IsNullOrWhiteSpace(lines[i])) continue;
                string[] values = lines[i].Split(',');
                if (values.Length < 4) continue;

                float thumb = float.Parse(values[values.Length - 4], CultureInfo.InvariantCulture);
                float index = float.Parse(values[values.Length - 3], CultureInfo.InvariantCulture);
                float middle = float.Parse(values[values.Length - 2], CultureInfo.InvariantCulture);
                float abduction = float.Parse(values[values.Length - 1], CultureInfo.InvariantCulture);

                _recordedFrames.Add(new float[] { thumb, index, middle, abduction });
                rowsRead++;
            }
            Debug.Log($"Dataset loaded: {_recordedFrames.Count} frames.");
        }

        private void ConfigureBody(ArticulationBody body)
        {
            ArticulationDrive xDrive = body.xDrive;
            xDrive.stiffness = stiffness; xDrive.damping = driveDamping; xDrive.forceLimit = forceLimit;
            body.xDrive = xDrive;
            body.mass = linkMass;
            if (body.jointType == ArticulationJointType.SphericalJoint) {
                ArticulationDrive yDrive = body.yDrive;
                yDrive.stiffness = stiffness; yDrive.damping = driveDamping; yDrive.forceLimit = forceLimit;
                body.yDrive = yDrive;
                ArticulationDrive zDrive = body.zDrive;
                zDrive.stiffness = stiffness; zDrive.damping = driveDamping; zDrive.forceLimit = forceLimit;
                body.zDrive = zDrive;
            }
        }

        void Update()
        {
            // dataset playback
            if (Input.GetKeyDown(KeyCode.Space))
            {
                isPlayingDataset = !isPlayingDataset;
                Debug.Log(isPlayingDataset ? "Motion Started" : "Motion Paused");
            }

            // input processing
            if (inputSource == ControlMode.Network)
                ProcessKeyboardInput();

            else if (inputSource == ControlMode.Dataset)
                ProcessDatasetInput();
            
            else if (inputSource == ControlMode.WeArt)
                ProcessWeArtInput();

            if (_hasData)
            {
                Vector<double> angles;
                lock (_lock) { 
                    angles = _incomingAngles.Clone(); 
                    _hasData = false; 
                }
                ApplyHumanPose(angles);
            }
            
            // Robot retargeting
            if (inputSource == ControlMode.Network || (inputSource == ControlMode.Dataset && isPlayingDataset))
                PerformRobotRetargeting();
            else
                UpdateHumanPrevPos();
        
        }

        private void PerformRobotRetargeting()
        {
            var v_h = Vector<double>.Build.Dense(h_refPoints.Length * 3);
            if (Time.deltaTime > 1e-5)
            {
                for (int i=0; i < h_refPoints.Length; i++)
                {
                    if (h_refPoints[i] == null) continue;
                    Vector3 curr = h_palm.InverseTransformPoint(h_refPoints[i].position);
                    Vector3 vel = (curr - _h_prevPos[i]) / Time.deltaTime;
                    v_h[i*3] = vel.x; v_h[i*3 + 1] = vel.y; v_h[i*3 + 2] = vel.z;
                }
            }

            Matrix<double> A_h = h_sphere.ComputeMatrixA(h_palm);
            Vector<double> sphereMotion = PInvTall(A_h) * v_h;

            Matrix<double> J_r = Kinematics.GetJacobian(r_joints, r_refPoints, r_palm, r_jointTypes);
            Matrix<double> J_r_pinv = PInv(J_r);
            Matrix<double> A_r = r_sphere.ComputeMatrixA(r_palm);

            float k = (h_sphere.Radius > 1e-5f) ? r_sphere.Radius / h_sphere.Radius : 1.0f;
            Matrix<double> K_c = Matrix<double>.Build.DenseIdentity(7, 7);
            K_c[0, 0] = k; K_c[1, 1] = k; K_c[2, 2] = k;

            Vector<double> v_r_local = A_r*(K_c*sphereMotion);
            Vector<double> v_r_world = Vector<double>.Build.Dense(v_r_local.Count);

            for (int i=0; i < r_refPoints.Length; i++)
            {
                if (r_refPoints[i] == null || i*3 + 2 >= v_r_local.Count) continue;
                Vector3 loc = new Vector3((float)v_r_local[i*3], (float)v_r_local[i*3 + 1], (float)v_r_local[i*3 + 2]);
                Vector3 wld = r_palm.TransformDirection(loc);
                v_r_world[i*3] = wld.x; v_r_world[i*3 + 1] = wld.y; v_r_world[i*3 + 2] = wld.z;
            }

            Vector<double> dq_particular = J_r_pinv * v_r_world;
            Vector<double> dq_final;
            if (useNullSpace)
            {
                Vector<double> dq_0 = ComputeGradient();
                Matrix<double> I = Matrix<double>.Build.DenseIdentity(_dof);
                Matrix<double> N = I - (J_r_pinv * J_r);
                dq_final = dq_particular + N*dq_0;
            }
            else dq_final = dq_particular;

            IntegrateVelocities(dq_final, Time.deltaTime);
        }

        void LateUpdate()
        {
            h_sphere.UpdateSphere(h_palm);
            r_sphere.UpdateSphere(r_palm);
            UpdateHumanPrevPos();
        }

        private void UpdateHumanPrevPos()
        {
            if (h_refPoints.Length > 0 && h_palm != null)
            {
                for (int i=0; i < h_refPoints.Length; i++)
                    if (h_refPoints[i] != null) _h_prevPos[i] = h_palm.InverseTransformPoint(h_refPoints[i].position);
            }
        }

        private void ProcessKeyboardInput() 
        {
            float dt = Time.deltaTime;
            if (Input.GetKey(KeyCode.T)) _synergyInput[0] += dt; 
            if (Input.GetKey(KeyCode.G)) _synergyInput[0] -= dt;
            if (Input.GetKey(KeyCode.I)) _synergyInput[1] += dt; 
            if (Input.GetKey(KeyCode.K)) _synergyInput[1] -= dt;
            if (Input.GetKey(KeyCode.M)) _synergyInput[2] += dt; 
            if (Input.GetKey(KeyCode.L)) _synergyInput[2] -= dt;
            if (Input.GetKey(KeyCode.A)) _synergyInput[3] += dt; 
            if (Input.GetKey(KeyCode.S)) _synergyInput[3] -= dt;
            for (int i = 0; i < 4; i++) _synergyInput[i] = Mathf.Clamp01(_synergyInput[i]);
        }

        private void ProcessDatasetInput()
        {
            if (isPlayingDataset && _recordedFrames.Count > 0)
            {
                _playbackTime += Time.deltaTime;
                _currentFrameIndex = Mathf.FloorToInt(_playbackTime * frameRate);

                if (_currentFrameIndex >= _recordedFrames.Count)
                     _currentFrameIndex = _recordedFrames.Count - 1;

                ApplyFrameToInput(_currentFrameIndex);
            }
        }

        private void ProcessWeArtInput()
        {
            // Input in tempo reale dai thimble WeArt
            if (thumbThimble != null)
            {
                _synergyInput[0] = thumbThimble.Closure.Value;
                _synergyInput[3] = thumbThimble.Abduction.Value;
            }
            if (indexThimble != null)
                _synergyInput[1] = indexThimble.Closure.Value;
            if (middleThimble != null)
                _synergyInput[2] = middleThimble.Closure.Value;
        }
        private void ApplyFrameToInput(int index)
        {
            if (index < 0 || index >= _recordedFrames.Count) return;
            float[] frame = _recordedFrames[index];
            if (frame.Length >= 4)
            {
                _synergyInput[0] = frame[0]; 
                _synergyInput[1] = frame[1]; 
                _synergyInput[2] = frame[2]; 
                _synergyInput[3] = frame[3]; 
            }
        }

        private void ApplyHumanPose(Vector<double> data) {
            if (data == null) return;
            for (int i=0; i < h_joints.Length; i++) {
                if (h_joints[i] == null) continue;
                int idx = i*3;
                h_joints[i].rotation = h_palm.rotation * Quaternion.Euler((float)data[idx], (float)data[idx + 1], (float)data[idx + 2]);
            }
        }

        private Vector<double> ComputeGradient()
        {
            var grad = Vector<double>.Build.Dense(_dof);
            double N = (double)_dof;
            
            int idx = 0;
            for (int i=0; i < r_joints.Length; i++)
            {
                if (r_joints[i] == null || i >= limitsMin.Length || i >= limitsMax.Length) continue;
                
                var type = r_jointTypes[i];
                var body = _r_bodies[i];

                // read current angles from physics (radians)
                float qX = (body != null && body.dofCount > 0) ? body.jointPosition[0] : 0;
                float qY = (body != null && body.dofCount > 1) ? body.jointPosition[1] : 0;
                float qZ = (body != null && body.dofCount > 2) ? body.jointPosition[2] : 0;
                
                // limits
                Vector3 min = limitsMin[i] * Mathf.Deg2Rad;
                Vector3 max = limitsMax[i] * Mathf.Deg2Rad;
                Vector3 range = max - min;
                Vector3 mid = (max + min) / 2.0f;

                // compute gradient elements
                if ((type == Kinematics.JointType.HingeX || type == Kinematics.JointType.HingeXY || type == Kinematics.JointType.Ball) && idx < grad.Count)
                {
                    double denom = range.x * range.x;
                    if (denom > 1e-9)
                        grad[idx] = (1.0/N) * (qX - mid.x) / denom;
                    idx++;
                }

                if ((type == Kinematics.JointType.HingeY || type == Kinematics.JointType.HingeXY || type == Kinematics.JointType.Ball) && idx < grad.Count)
                {
                    float val = (body.jointType == ArticulationJointType.RevoluteJoint) ? qX : qY;
                    double denom = range.y * range.y;
                    if (denom > 1e-9) 
                        grad[idx] = (1.0/N) * (val - mid.y) / denom;
                    idx++;
                }

                if ((type == Kinematics.JointType.HingeZ || type == Kinematics.JointType.Ball) && idx < grad.Count)
                {
                    float val = (body.jointType == ArticulationJointType.RevoluteJoint) ? qX : qZ;
                    double denom = range.z * range.z;
                    if (denom > 1e-9) 
                        grad[idx] = (1.0/N) * (val - mid.z) / denom;
                    idx++;
                }
            }
            
            // q0 = -eta * grad
            return -nullSpaceGain * grad;
        }

        private void NetworkLoop() {
            byte[] sendBuf = new byte[16]; byte[] recvBuf = new byte[45 * 4]; float[] floats = new float[45]; 
            while (_active) { 
                try { 
                    _client = new TcpClient(); 
                    _client.Connect(ipAddress, port); 
                    _stream = _client.GetStream(); 
                    while (_active && _client.Connected) { 
                        Buffer.BlockCopy(_synergyInput, 0, sendBuf, 0, 16); 
                        _stream.Write(sendBuf, 0, 16); 
                        int read = 0; 
                        while (read < recvBuf.Length && _active) { 
                            int chunk = _stream.Read(recvBuf, read, recvBuf.Length - read); 
                            if (chunk == 0) throw new Exception("Disconnect"); 
                            read += chunk; 
                        } 
                        Buffer.BlockCopy(recvBuf, 0, floats, 0, recvBuf.Length); 
                        var vec = Vector<double>.Build.DenseOfArray(Array.ConvertAll(floats, x => (double)x)); 
                        lock (_lock) { 
                            _incomingAngles = vec; 
                            _hasData = true; 
                        } 
                    } 
                } catch { 
                    if (_active) Thread.Sleep(2000); 
                } finally { 
                    _stream?.Close(); _client?.Close(); 
                } 
            } 
        }

        private void IntegrateVelocities(Vector<double> dq, float dt) {
            if (dq == null || dq.Count != _dof) return;
            int idx = 0;

            for (int i=0; i < r_joints.Length; i++) {
                if (r_joints[i] == null) continue;

                var body = _r_bodies[i];
                var type = r_jointTypes[i];

                // get velocity for this joint
                float velocity = (float)dq[idx];

                // increment axis based on DoF
                if (type == Kinematics.JointType.HingeXY) idx += 2;
                else if (type == Kinematics.JointType.Ball) idx += 3;
                else idx += 1;

                if (body == null) continue;

                if (body.jointType == ArticulationJointType.RevoluteJoint) {
                    // URDF importer aligns rotation axis to xDrive
                    float deltaDeg = velocity * velocityGain * Mathf.Rad2Deg * dt;

                    var drive = body.xDrive;
                    drive.target += deltaDeg;

                    // clamp within limits
                    //if (drive.upperLimit > drive.lowerLimit)
                    //    drive.target = Mathf.Clamp(drive.target, drive.lowerLimit, drive.upperLimit);

                    body.xDrive = drive;
                }
            }
        }

        private Matrix<double> PInv(Matrix<double> M) {
            var Mt = M.Transpose(); 
            var I = Matrix<double>.Build.DenseIdentity(M.RowCount); 
            return Mt * (M * Mt + damping*damping*I).Inverse(); 
        }
        
        private Matrix<double> PInvTall(Matrix<double> M) { 
            var Mt = M.Transpose(); 
            var I = Matrix<double>.Build.DenseIdentity(M.ColumnCount); 
            return (Mt * M + damping*damping*I).Inverse() * Mt; 
        }

        void AssignSphereRefs() { 
            if (h_sphere) h_sphere.referencePoints = h_refPoints; 
            if (r_sphere) r_sphere.referencePoints = r_refPoints; 
        }

        void OnApplicationQuit() { 
            _active = false; 
            _stream?.Close(); 
            _client?.Close(); 
            if (_netThread != null && _netThread.IsAlive) 
                _netThread.Join(500); 
        }
    }
}

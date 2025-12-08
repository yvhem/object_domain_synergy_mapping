using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Text;
using WeArt.Components;

public class Test : MonoBehaviour
{
    [Header("References")]
    [Tooltip("Human Virtual Sphere")]
    public VirtualSphere h_sphere;
    [Tooltip("Robot Virtual Sphere")]
    public VirtualSphere r_sphere;

    [Header("Validation Settings")]
    [Tooltip("1.0 for normalized comparison.")]
    public float springStiffness = 1.0f;
    [Tooltip("File name for the CSV report.")]
    public string fileName = "GraspValidationReport.csv";
    [SerializeField]
    public string saveDirectory;

    // Internal state
    private bool _isRecording = false;
    private float _initialRadiusH;
    private float _initialRadiusR;
    private float _scalingFactor;

    // Data storage for CSV
    private StringBuilder _csvContent = new StringBuilder();

    void Start()
    {
        // Initialize CSV header
        _csvContent.AppendLine("Time,HumanRadius,RobotRadius,ScalingFactor,HumanEnergy,RobotEnergy,"+
                                "hPos_x,hPos_y,hPos_z,rPos_x,rPos_y,rPos_z");
    }

    void Update()
    {
        // Controls to Start/Stop Recording
        if (Input.GetKeyDown(KeyCode.R))
        {
            StartRecording();
        }
        if (Input.GetKeyDown(KeyCode.S))
        {
            StopRecording();
        }

        if (_isRecording)
        {
            PerformValidationStep();
        }
    }


    void StartRecording()
    {
        if (h_sphere == null || r_sphere == null)
        {
            Debug.LogError("Validation Error: Please assign both Human and Robot Virtual Spheres in the Inspector.");
            return;
        }

        _initialRadiusH = h_sphere.Radius;
        _initialRadiusR = r_sphere.Radius;

        if (_initialRadiusH > 0)
            _scalingFactor = _initialRadiusR / _initialRadiusH;
        else
            _scalingFactor = 1.0f;

        _csvContent.Clear();
        _csvContent.AppendLine("Time,HumanRadius,RobotRadius,ScalingFactor,HumanEnergy,RobotEnergy,"+
                                "hPos_x,hPos_y,hPos_z,rPos_x,rPos_y,rPos_z");


        _isRecording = true;
        Debug.Log($"<color=green>VALIDATION STARTED.</color> Baseline H: {_initialRadiusH:F4}, Baseline R: {_initialRadiusR:F4}, Scaling Factor: {_scalingFactor:F4}");
    }

    void StopRecording()
    {
        _isRecording = false;

        string directory = saveDirectory;

        if (string.IsNullOrEmpty(directory))
        {
            Debug.LogError("Save directory is not set in the Inspector.");
            return;
        }

        // Crea la cartella se non esiste
        if (!Directory.Exists(directory))
            Directory.CreateDirectory(directory);

        string filePath = Path.Combine(directory, fileName);

        try
        {
            File.WriteAllText(filePath, _csvContent.ToString());
            Debug.Log($"<color=red>VALIDATION STOPPED.</color> Report salvato in: {filePath}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to save file: {e.Message}");
        }
    }

    void PerformValidationStep()
    {
        // Current radii
        float currentRadiusH = h_sphere.Radius;
        float currentRadiusR = r_sphere.Radius;

        // Compute Deltas (normalized)
        float deformationH = Mathf.Abs(_initialRadiusH - currentRadiusH);
        float deformationR = Mathf.Abs(_initialRadiusR - currentRadiusR);
        float normDefH = deformationH / _initialRadiusH;
        float normDefR = deformationR / _initialRadiusR;

        // Compute Energy (normalized)
        float _ncpH = h_sphere.referencePoints.Length;
        float _ncpR = r_sphere.referencePoints.Length;
        float cp_scalingFactor= _ncpH / _ncpR;
        float energyHnorm = 0.5f  * springStiffness * normDefH * normDefH;
        float energyRnorm = 0.5f * cp_scalingFactor * springStiffness * normDefR * normDefR;

        // Position
        Vector3 hPos = h_sphere.transform.position;
        Vector3 rPos = r_sphere.transform.position;

        // Log data
        string line = $"{Time.time},{currentRadiusH},{currentRadiusR},{_scalingFactor},{energyHnorm},{energyRnorm}," +
                    $"{hPos.x},{hPos.y},{hPos.z},{rPos.x},{rPos.y},{rPos.z}";

        _csvContent.AppendLine(line);
    }

    // Visual Debugging
    void OnGUI()
    {
        if (_isRecording)
        {
            GUI.Label(new Rect(10, 10, 300, 20), $"Recording Validation... (Press S to Stop)");
        }
        else
        {
            GUI.Label(new Rect(10, 10, 300, 20), "Press R to Start Validation Recording");
        }
    }
}
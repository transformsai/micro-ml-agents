using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class ReacherAcademy : Academy {

    public float goalSize;
    public float goalSpeed;


    public override void AcademyReset()
    {
        goalSize = (float)resetParameters["goal_size"];
        goalSpeed = (float)resetParameters["goal_speed"];
        Physics.gravity = new Vector3(0, -resetParameters["gravity"], 0);
    }

    public override void AcademyStep()
    {


    }

}

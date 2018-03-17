#!/usr/bin/env python
import rospy
from dummy_detector import DummyDetector
from live_detector import TLDetector

detector = 'dummy'

if __name__ == '__main__':
    try:
        if detector == 'dummy':
            DummyDetector()
        else:
            TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')

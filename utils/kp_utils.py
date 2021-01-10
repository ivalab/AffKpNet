def read_kp_profile(filepath):
    aff, offset, keypoint_list, aff2, offset2, keypoint_list2 = None, [],[], None, [],[]
    with open(filepath) as fp:
        # read affordance
        line = fp.readline()
        aff = line.strip()

        # read offset
        line = fp.readline()
        elements = line.split()
        for element in elements[1:]:
            offset.append(float(element))
        offset.append(0.0)

        # read keypont_list
        for count in xrange(5):
            line = fp.readline()
            elements = line.split()
            keypoint = []
            for element in elements[1:]:
                keypoint.append(float(element))
            keypoint.append(1.0)
            keypoint_list.append(keypoint)

        line = fp.readline()

        if line:
            aff2 = line.strip()

            # read offset
            line = fp.readline()
            elements = line.split()
            for element in elements[1:]:
                offset2.append(float(element))
            offset2.append(0.0)

            # read keypont_list
            for count in xrange(5):
                line = fp.readline()
                elements = line.split()
                keypoint = []
                for element in elements[1:]:
                    keypoint.append(float(element))
                keypoint.append(1.0)
                keypoint_list2.append(keypoint)


        return aff, offset, keypoint_list, aff2, offset2, keypoint_list2

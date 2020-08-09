''' 
Based on the script provided by Jun Zhou
Junjie Cao
'''

import numpy as np
import trimesh


def mesh2points(mesh, count):
    """
    given a 3d mesh and return the uniform samples points on the surface
    input: a 3d mesh
    the number of the sample points
    output: points of the 3d mesh, id, normals
    """
    # of each face of the mesh
    area = mesh.area_faces
    # total area (float)
    area_sum = np.sum(area)
    # cumulative area (len(mesh.faces))
    area_cum = np.cumsum(area)
    face_pick = np.random.random(count) * area_sum
    face_index = np.searchsorted(area_cum, face_pick)

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = mesh.triangles[:, 0]
    tri_vectors = mesh.triangles[:, 1:].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = np.random.random((len(tri_vectors), 2, 1))

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)

    normals = mesh.face_normals
    new_normals = normals[face_index, :]

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    return samples, face_index, new_normals

def test():
    # attach to logger so trimesh messages will be printed to console
    trimesh.util.attach_to_log()

    mesh = trimesh.load('test.obj', process=False)
    mesh.vertices.min(), mesh.vertices.max() # -1 -- +1

    #mesh.show()
    # v= np.array(mesh.vertices)
    # mesh1 = trimesh.Trimesh(v, mesh.faces)
    # mesh1.show()
    [samples, face_index, new_normals] = mesh2points(mesh, 10240)
    samples = trimesh.PointCloud(samples)
    scene = mesh.scene()
    scene.add_geometry(samples)
    #scene.show()
    
    try:
        # increment the file name
        file_name = 'render_' + '.png'
        # save a render of the object as a png
        png = scene.save_image(resolution=[800, 600],visible=True)
        with open(file_name, 'wb') as f:
            f.write(png)
            f.close()

    except BaseException as E:
        print("unable to save image", str(E))


if __name__ == '__main__':
    import os
    import json
    test()

    DATA_ROOT = '../../../data/spe_data_2500' 

    jsonfile = os.path.join(DATA_ROOT, 'data.json')
    json_file = open(jsonfile)
    meshpath =  os.path.join(DATA_ROOT, 'mesh')
    imgpath =  os.path.join(DATA_ROOT, 'image')
    outpath = os.path.join(DATA_ROOT, 'point')
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    if not os.path.exists(imgpath):
        os.makedirs(imgpath)

    outcount = 0
    #scene = trimesh.scene.Scene()
    #viewer = scene.show()
    #json_file = open(jsonfile)
    with open(jsonfile) as json_file:  
        data = json.load(json_file)
        targetnpts = data['count_resample']
        strnpts = str(targetnpts)
        for p in data['people']:
            name = p['name']
            filename = os.path.join(meshpath, name + '.obj')  
            outfilename = os.path.join(outpath, name + '_pts_' + strnpts + '.pts')
            imfile = os.path.join(imgpath, name + '_img_' + strnpts + '.png')

            ########## 
            mesh = trimesh.load(filename, process=False)
            samples, _, _ = mesh2points(mesh, targetnpts)
            with open(outfilename, 'w') as fp:
                for v in samples:
                    fp.write('%f %f %f\n' % (v[0], v[1], v[2]))  
                fp.close()
                    
            ########## save image
            scene = mesh.scene()
            scene.add_geometry(trimesh.PointCloud(samples))
            # if outcount == 0:
            #     viewer = scene.show()
            #     viewer.set_location(10, 10)
            
            try:
                # save a render of the object as a png
                png = scene.save_image(resolution=[320, 320], visible=True)
                with open(imfile, 'wb') as f:
                    f.write(png)
                    f.close()

            except BaseException as E:
                print("unable to save image", str(E))  
            
            ########## 
            print(outcount)
            outcount +=1
        json_file.close()
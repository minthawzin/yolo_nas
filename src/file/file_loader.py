import os
from src.file import movie_loader

class FileLoader:

    @property
    def filePath( self ):
        return self.__file_path

    @property
    def fileExtension( self ):
        return self.__extension

    @property
    def validFileTypes( self ):
        return self.__validFileTypes

    @property
    def movieLoader( self ):
        return self.__movieLoader

    @property
    def movieName( self ):
        return self.filePath.split('/')[-1].split('.')[0]

    def __init__( self, file_path, start_time="2021:06:14.09:00:00" ):
        self.__file_path      = file_path
        self.__extension      =  os.path.splitext(self.filePath)[1]
        self.__validFileTypes = [".MP4", ".mp4", ".AVI", ".avi", ".MOV", ".MTS"]
        self.__movieLoader    = movie_loader.MovieLoader( self.filePath, start_time, self.movieName )
        self.baseName         = self.filePath.split("/")[-1].split('.')[0]

        self.__checkValidFileTypes()

    def __checkValidFileTypes( self ):
        if self.fileExtension not in self.validFileTypes:
            raise ValueError("Invalid file Type")
